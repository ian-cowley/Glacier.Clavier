namespace Glacier.Clavier.Engine;

using System;
using System.Buffers;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Clavier.Model;

/// <summary>
/// Delegate for extracting state representation vectors into caller-provided float spans.
/// Compatible with Glacier.Inference, ONNX, and custom embedding models.
/// </summary>
public delegate void ClavierStateEncoder(ReadOnlySpan<char> text, Span<float> destination);

/// <summary>
/// High-throughput execution session for sub-millisecond System-1 discrete policy inference, online calibration training, and checkpoint persistence.
/// Coordinates state &amp; candidate representations with lock-free concurrent forward passes.
/// </summary>
public sealed class ClavierSession : IDisposable
{
    private readonly ClavierDecisionHead _decisionHead;
    private readonly ClavierNoulHead _noulHead;
    private readonly ClavierScoreHead _scoreHead;
    private readonly ClavierStateEncoder? _encoder;
    private readonly int _embeddingDim;
    private readonly int _hiddenDim;
    private readonly int _numActions;
    private readonly ReaderWriterLockSlim _rwLock = new(LockRecursionPolicy.NoRecursion);
    private volatile bool _disposed;

    public int EmbeddingDim => _embeddingDim;
    public int HiddenDim => _hiddenDim;
    public int NumActions => _numActions;

    public ClavierDecisionHead DecisionHead => _decisionHead;
    public ClavierNoulHead NoulHead => _noulHead;
    public ClavierScoreHead ScoreHead => _scoreHead;

        public float Temperature
    {
        get => _decisionHead.Temperature;
        set
        {
            _rwLock.EnterWriteLock();
            try
            {
                _decisionHead.Temperature = value;
                _noulHead.Temperature = value;
                _scoreHead.Temperature = value;
            }
            finally
            {
                _rwLock.ExitWriteLock();
            }
        }
    }

    private float _conformalQuantile = 0.95f;
    private float _conformalAlpha = 0.05f;
    private float[]? _conformalCalibrationScores;

    /// <summary>
    /// Calibrated conformal non-conformity quantile threshold qHat in [0.0, 1.0].
    /// Candidates are admitted to C(x) when 1 - P(y|x) &lt;= qHat &lt;=&gt; P(y|x) &gt;= 1 - qHat.
    /// </summary>
    public float ConformalQuantile
    {
        get => _conformalQuantile;
        set => _conformalQuantile = Math.Clamp(value, 0.0f, 1.0f);
    }

    /// <summary>
    /// Default significance level alpha in (0, 1). Default is 0.05 (95% marginal coverage).
    /// </summary>
    public float ConformalAlpha
    {
        get => _conformalAlpha;
        set
        {
            if (value <= 0.0f || value >= 1.0f || float.IsNaN(value))
                throw new ArgumentOutOfRangeException(nameof(value), "Alpha must be in (0, 1).");
            _conformalAlpha = value;
            if (_conformalCalibrationScores != null && _conformalCalibrationScores.Length > 0)
            {
                _conformalQuantile = ClavierCalibration.CalibrateQuantile(_conformalCalibrationScores, value);
            }
        }
    }

    /// <summary>
    /// Calibrates the session's conformal quantile threshold qHat from holdout non-conformity scores.
    /// </summary>
    public void CalibrateConformal(ReadOnlySpan<float> calibrationScores, float alpha = 0.05f)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        _rwLock.EnterWriteLock();
        try
        {
            _conformalCalibrationScores = calibrationScores.ToArray();
            _conformalAlpha = alpha;
            _conformalQuantile = ClavierCalibration.CalibrateQuantile(calibrationScores, alpha);
        }
        finally
        {
            _rwLock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Evaluates environment state against actions and constructs Split Conformal Prediction Set C(x).
    /// Returns detailed prediction set with System-1 fast dispatch vs System-2 LLM escalation gating.
    /// </summary>
    public ClavierPredictionSet DecideConformalSet(
        ReadOnlySpan<char> stateText,
        float alpha = 0.05f,
        ReadOnlySpan<string> candidateActions = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var sw = Stopwatch.StartNew();
        int activeActions = !candidateActions.IsEmpty ? candidateActions.Length : _numActions;
        var probs = new float[activeActions];

        ClavierDecision decision;
        float qHat;

        _rwLock.EnterReadLock();
        try
        {
            decision = DecideCore(stateText, candidateActions, probs);
            qHat = (_conformalCalibrationScores != null && _conformalCalibrationScores.Length > 0 && MathF.Abs(alpha - _conformalAlpha) > 1e-5f)
                ? ClavierCalibration.CalibrateQuantile(_conformalCalibrationScores, alpha)
                : _conformalQuantile;
        }
        finally
        {
            _rwLock.ExitReadLock();
        }

        float tau = 1.0f - qHat; // Minimum probability threshold for inclusion
        ulong actionMask = 0UL;
        int count = 0;

        uint[]? rentedMatched = null;
        Span<uint> matchedSpan = activeActions <= 256
            ? stackalloc uint[activeActions]
            : (rentedMatched = ArrayPool<uint>.Shared.Rent(activeActions)).AsSpan(0, activeActions);

        uint[] actionIds;
        try
        {
            for (int a = 0; a < activeActions; a++)
            {
                if (probs[a] >= tau)
                {
                    matchedSpan[count++] = (uint)a;
                    if (a < 64)
                    {
                        actionMask |= (1UL << a);
                    }
                }
            }

            actionIds = count > 0 ? matchedSpan[..count].ToArray() : [];
        }
        finally
        {
            if (rentedMatched != null)
            {
                ArrayPool<uint>.Shared.Return(rentedMatched);
            }
        }

        string[]? actionNames = null;
        if (!candidateActions.IsEmpty)
        {
            actionNames = new string[count];
            for (int i = 0; i < count; i++)
            {
                uint id = actionIds[i];
                actionNames[i] = id < (uint)candidateActions.Length ? candidateActions[(int)id] : id.ToString();
            }
        }

        string? bestActionName = !candidateActions.IsEmpty && decision.ActionId < (uint)candidateActions.Length
            ? candidateActions[(int)decision.ActionId]
            : null;

        ConformalEscalationReason escalation = count switch
        {
            1 => ConformalEscalationReason.None,
            0 => ConformalEscalationReason.EmptyPredictionSet,
            _ => ConformalEscalationReason.AmbiguousPredictionSet
        };

        sw.Stop();

        return new ClavierPredictionSet
        {
            Count = count,
            EscalationReason = escalation,
            BestActionId = decision.ActionId,
            BestActionName = bestActionName,
            BestConfidence = decision.Confidence,
            ActionIds = actionIds,
            ActionNames = actionNames,
            ActionMask = actionMask,
            QuantileThreshold = qHat,
            InclusionProbabilityThreshold = tau,
            Alpha = alpha,
            Probabilities = probs,
            LatencyMs = sw.Elapsed.TotalMilliseconds
        };
    }

    public ClavierSession(
        int embeddingDim = 768,
        int hiddenDim = 256,
        int numActions = 64,
        float temperature = 1.0f,
        ClavierStateEncoder? encoder = null)
    {
        if (embeddingDim <= 0 || hiddenDim <= 0 || numActions <= 0)
            throw new ArgumentException("Dimensions and action counts must be positive.");

        _embeddingDim = embeddingDim;
        _hiddenDim = hiddenDim;
        _numActions = numActions;
        _encoder = encoder;
        _decisionHead = new ClavierDecisionHead(embeddingDim, hiddenDim, numActions, temperature);
        _noulHead = new ClavierNoulHead(embeddingDim, hiddenDim, temperature);
        _scoreHead = new ClavierScoreHead(embeddingDim, hiddenDim, temperature);
    }

    /// <summary>
    /// Evaluates environment state against candidate actions in a single forward pass.
    /// When candidateActions is non-empty, each candidate string is jointly encoded with stateText so action semantics directly determine logits.
    /// </summary>
    public ClavierDecision Decide(ReadOnlySpan<char> stateText, ReadOnlySpan<string> candidateActions = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        _rwLock.EnterReadLock();
        try
        {
            return DecideCore(stateText, candidateActions, Span<float>.Empty);
        }
        finally
        {
            _rwLock.ExitReadLock();
        }
    }

    /// <summary>
    /// Evaluates environment state and returns full calibrated probability distribution across actions.
    /// </summary>
    public ClavierDecisionResult DecideWithDistribution(ReadOnlySpan<char> stateText, ReadOnlySpan<string> candidateActions = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var sw = Stopwatch.StartNew();
        int activeActions = !candidateActions.IsEmpty ? candidateActions.Length : _numActions;
        var probs = new float[activeActions];

        ClavierDecision decision;
        _rwLock.EnterReadLock();
        try
        {
            decision = DecideCore(stateText, candidateActions, probs);
        }
        finally
        {
            _rwLock.ExitReadLock();
        }
        sw.Stop();

        string? actionName = !candidateActions.IsEmpty && decision.ActionId < (uint)candidateActions.Length
            ? candidateActions[(int)decision.ActionId]
            : null;

        return new ClavierDecisionResult
        {
            Decision = decision,
            ActionName = actionName,
            Probabilities = probs,
            LatencyMs = sw.Elapsed.TotalMilliseconds
        };
    }

    private ClavierDecision DecideCore(
        ReadOnlySpan<char> stateText,
        ReadOnlySpan<string> candidateActions,
        Span<float> probabilitiesOut)
    {
        float[]? rentedEmb = null;
        float[]? rentedHidden = null;
        float[]? rentedLogits = null;

        int actionCount = !candidateActions.IsEmpty ? candidateActions.Length : _numActions;
        Span<float> emb = _embeddingDim <= 1024
            ? stackalloc float[_embeddingDim]
            : (rentedEmb = ArrayPool<float>.Shared.Rent(_embeddingDim)).AsSpan(0, _embeddingDim);
        Span<float> hidden = _hiddenDim <= 512
            ? stackalloc float[_hiddenDim]
            : (rentedHidden = ArrayPool<float>.Shared.Rent(_hiddenDim)).AsSpan(0, _hiddenDim);
        Span<float> logits = actionCount <= 256
            ? stackalloc float[actionCount]
            : (rentedLogits = ArrayPool<float>.Shared.Rent(actionCount)).AsSpan(0, actionCount);

        try
        {
            if (candidateActions.IsEmpty)
            {
                ExtractStateEmbedding(stateText, emb);
                return _decisionHead.Forward(emb, hidden, logits, probabilitiesOut, _numActions);
            }

            float maxLogit = float.NegativeInfinity;
            for (int a = 0; a < candidateActions.Length; a++)
            {
                ExtractCombinedEmbedding(stateText, candidateActions[a].AsSpan(), emb);
                float s = _decisionHead.ForwardCandidateLogit(emb, hidden);
                logits[a] = s;
                if (s > maxLogit)
                {
                    maxLogit = s;
                }
            }

            return ClavierDecisionHead.SoftmaxAndArgmax(logits, probabilitiesOut, maxLogit);
        }
        finally
        {
            if (rentedEmb != null) ArrayPool<float>.Shared.Return(rentedEmb);
            if (rentedHidden != null) ArrayPool<float>.Shared.Return(rentedHidden);
            if (rentedLogits != null) ArrayPool<float>.Shared.Return(rentedLogits);
        }
    }

    /// <summary>
    /// Batch-evaluates multiple independent environment states concurrently without lock contention.
    /// </summary>
    public void DecideBatch(
        ReadOnlySpan<ReadOnlyMemory<char>> states,
        Span<ClavierDecision> decisions,
        ReadOnlySpan<string> candidateActions = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        int count = Math.Min(states.Length, decisions.Length);
        if (count == 0) return;

        if (count >= 16)
        {
            var stateArr = states.Slice(0, count).ToArray();
            var candArr = candidateActions.IsEmpty ? Array.Empty<string>() : candidateActions.ToArray();
            var outArr = new ClavierDecision[count];
            Parallel.For(0, count, i =>
            {
                outArr[i] = Decide(stateArr[i].Span, candArr);
            });
            outArr.AsSpan().CopyTo(decisions);
            return;
        }

        for (int i = 0; i < count; i++)
        {
            decisions[i] = Decide(states[i].Span, candidateActions);
        }
    }

    /// <summary>
    /// Evaluates an assertion or hypothesis against an environment state in a single forward pass (Jev 'Noul' primitive).
    /// </summary>
    public ClavierNoul Verify(ReadOnlySpan<char> stateText, ReadOnlySpan<char> hypothesis = default, float threshold = 0.5f)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        float[]? rentedEmb = null;
        float[]? rentedHidden = null;
        Span<float> emb = _embeddingDim <= 1024
            ? stackalloc float[_embeddingDim]
            : (rentedEmb = ArrayPool<float>.Shared.Rent(_embeddingDim)).AsSpan(0, _embeddingDim);
        Span<float> hidden = _hiddenDim <= 512
            ? stackalloc float[_hiddenDim]
            : (rentedHidden = ArrayPool<float>.Shared.Rent(_hiddenDim)).AsSpan(0, _hiddenDim);

        _rwLock.EnterReadLock();
        try
        {
            ExtractCombinedEmbedding(stateText, hypothesis, emb);
            return _noulHead.Forward(emb, hidden, threshold);
        }
        finally
        {
            _rwLock.ExitReadLock();
            if (rentedEmb != null) ArrayPool<float>.Shared.Return(rentedEmb);
            if (rentedHidden != null) ArrayPool<float>.Shared.Return(rentedHidden);
        }
    }

    /// <summary>
    /// Evaluates an assertion or hypothesis with latency timing and calibrated confidence metrics.
    /// </summary>
    public ClavierNoulResult VerifyWithDetails(ReadOnlySpan<char> stateText, ReadOnlySpan<char> hypothesis = default, float threshold = 0.5f)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var sw = Stopwatch.StartNew();
        ClavierNoul noul = Verify(stateText, hypothesis, threshold);
        sw.Stop();

        return new ClavierNoulResult
        {
            Noul = noul,
            IsAffirmative = noul.IsAffirmative,
            Probability = noul.Probability,
            Confidence = noul.Confidence,
            LatencyMs = sw.Elapsed.TotalMilliseconds
        };
    }

    /// <summary>
    /// Batch-evaluates multiple assertions / states concurrently.
    /// </summary>
    public void VerifyBatch(
        ReadOnlySpan<ReadOnlyMemory<char>> states,
        Span<ClavierNoul> results,
        ReadOnlySpan<char> hypothesis = default,
        float threshold = 0.5f)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        int count = Math.Min(states.Length, results.Length);
        if (count == 0) return;

        if (count >= 16)
        {
            var stateArr = states.Slice(0, count).ToArray();
            string hypStr = hypothesis.ToString();
            var outArr = new ClavierNoul[count];
            Parallel.For(0, count, i =>
            {
                outArr[i] = Verify(stateArr[i].Span, hypStr.AsSpan(), threshold);
            });
            outArr.AsSpan().CopyTo(results);
            return;
        }

        for (int i = 0; i < count; i++)
        {
            results[i] = Verify(states[i].Span, hypothesis, threshold);
        }
    }

    /// <summary>
    /// Evaluates continuous scalar rating against criteria on scale [min, max] in a single forward pass (Jev 'Score' primitive).
    /// </summary>
    public ClavierScore Score(
        ReadOnlySpan<char> stateText,
        ReadOnlySpan<char> criteria = default,
        float min = 0.0f,
        float max = 1.0f)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        float[]? rentedEmb = null;
        float[]? rentedHidden = null;
        Span<float> emb = _embeddingDim <= 1024
            ? stackalloc float[_embeddingDim]
            : (rentedEmb = ArrayPool<float>.Shared.Rent(_embeddingDim)).AsSpan(0, _embeddingDim);
        Span<float> hidden = _hiddenDim <= 512
            ? stackalloc float[_hiddenDim]
            : (rentedHidden = ArrayPool<float>.Shared.Rent(_hiddenDim)).AsSpan(0, _hiddenDim);

        _rwLock.EnterReadLock();
        try
        {
            ExtractCombinedEmbedding(stateText, criteria, emb);
            return _scoreHead.Forward(emb, hidden, min, max);
        }
        finally
        {
            _rwLock.ExitReadLock();
            if (rentedEmb != null) ArrayPool<float>.Shared.Return(rentedEmb);
            if (rentedHidden != null) ArrayPool<float>.Shared.Return(rentedHidden);
        }
    }

    /// <summary>
    /// Evaluates continuous scalar rating with scale bounds and latency details.
    /// </summary>
    public ClavierScoreResult ScoreWithDetails(
        ReadOnlySpan<char> stateText,
        ReadOnlySpan<char> criteria = default,
        float min = 0.0f,
        float max = 1.0f)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var sw = Stopwatch.StartNew();
        ClavierScore score = Score(stateText, criteria, min, max);
        sw.Stop();

        return new ClavierScoreResult
        {
            Score = score,
            Value = score.Value,
            Confidence = score.Confidence,
            Min = min,
            Max = max,
            LatencyMs = sw.Elapsed.TotalMilliseconds
        };
    }

    /// <summary>
    /// Batch-evaluates multiple scalar scores across states.
    /// </summary>
    public void ScoreBatch(
        ReadOnlySpan<ReadOnlyMemory<char>> states,
        Span<ClavierScore> results,
        ReadOnlySpan<char> criteria = default,
        float min = 0.0f,
        float max = 1.0f)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        int count = Math.Min(states.Length, results.Length);
        if (count == 0) return;

        if (count >= 16)
        {
            var stateArr = states.Slice(0, count).ToArray();
            string critStr = criteria.ToString();
            var outArr = new ClavierScore[count];
            Parallel.For(0, count, i =>
            {
                outArr[i] = Score(stateArr[i].Span, critStr.AsSpan(), min, max);
            });
            outArr.AsSpan().CopyTo(results);
            return;
        }

        for (int i = 0; i < count; i++)
        {
            results[i] = Score(states[i].Span, criteria, min, max);
        }
    }

    /// <summary>
    /// Trains the Choice decision head on a supervised (state, targetActionIndex) example.
    /// Supports both fixed action slots (candidateActions empty) and dynamic candidate lists.
    /// </summary>
    public float TrainDecision(
        ReadOnlySpan<char> stateText,
        int targetActionIndex,
        ReadOnlySpan<string> candidateActions = default,
        float learningRate = 0.05f,
        float labelSmoothing = 0.0f)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        _rwLock.EnterWriteLock();
        try
        {
            if (candidateActions.IsEmpty)
            {
                float[]? rented = null;
                Span<float> emb = _embeddingDim <= 1024
                    ? stackalloc float[_embeddingDim]
                    : (rented = ArrayPool<float>.Shared.Rent(_embeddingDim)).AsSpan(0, _embeddingDim);
                try
                {
                    ExtractStateEmbedding(stateText, emb);
                    return _decisionHead.TrainStep(emb, targetActionIndex, learningRate, labelSmoothing);
                }
                finally
                {
                    if (rented != null) ArrayPool<float>.Shared.Return(rented);
                }
            }

            int k = candidateActions.Length;
            float[] jointPool = ArrayPool<float>.Shared.Rent(k * _embeddingDim);
            try
            {
                for (int a = 0; a < k; a++)
                {
                    ExtractCombinedEmbedding(stateText, candidateActions[a].AsSpan(), jointPool.AsSpan(a * _embeddingDim, _embeddingDim));
                }
                return _decisionHead.TrainCandidateStep(jointPool.AsSpan(0, k * _embeddingDim), k, targetActionIndex, learningRate);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(jointPool);
            }
        }
        finally
        {
            _rwLock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Trains the Noul binary verification head on a supervised (state, hypothesis, targetAffirmative) example.
    /// </summary>
    public float TrainVerify(
        ReadOnlySpan<char> stateText,
        ReadOnlySpan<char> hypothesis,
        bool targetAffirmative,
        float learningRate = 0.05f)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        float[]? rented = null;
        Span<float> emb = _embeddingDim <= 1024
            ? stackalloc float[_embeddingDim]
            : (rented = ArrayPool<float>.Shared.Rent(_embeddingDim)).AsSpan(0, _embeddingDim);

        _rwLock.EnterWriteLock();
        try
        {
            ExtractCombinedEmbedding(stateText, hypothesis, emb);
            return _noulHead.TrainStep(emb, targetAffirmative, learningRate);
        }
        finally
        {
            _rwLock.ExitWriteLock();
            if (rented != null) ArrayPool<float>.Shared.Return(rented);
        }
    }

    /// <summary>
    /// Trains the Score continuous rating head on a supervised (state, criteria, targetValue) example.
    /// </summary>
    public float TrainScore(
        ReadOnlySpan<char> stateText,
        ReadOnlySpan<char> criteria,
        float targetValue,
        float min = 0.0f,
        float max = 1.0f,
        float learningRate = 0.05f)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        float[]? rented = null;
        Span<float> emb = _embeddingDim <= 1024
            ? stackalloc float[_embeddingDim]
            : (rented = ArrayPool<float>.Shared.Rent(_embeddingDim)).AsSpan(0, _embeddingDim);

        _rwLock.EnterWriteLock();
        try
        {
            ExtractCombinedEmbedding(stateText, criteria, emb);
            return _scoreHead.TrainStep(emb, targetValue, min, max, learningRate);
        }
        finally
        {
            _rwLock.ExitWriteLock();
            if (rented != null) ArrayPool<float>.Shared.Return(rented);
        }
    }

    /// <summary>
    /// Saves all three calibrated head weights, temperature settings, and conformal parameters to a binary checkpoint stream.
    /// </summary>
    public void SaveCheckpoint(Stream stream)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(stream);

        _rwLock.EnterReadLock();
        try
        {
            using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
            writer.Write(0x52564C43u); // "CLVR"
            writer.Write(2);           // format version 2
            writer.Write(_embeddingDim);
            writer.Write(_hiddenDim);
            writer.Write(_numActions);
            writer.Write(_conformalQuantile);
            writer.Write(_conformalAlpha);
            int calCount = _conformalCalibrationScores?.Length ?? 0;
            writer.Write(calCount);
            if (calCount > 0 && _conformalCalibrationScores != null)
            {
                for (int i = 0; i < calCount; i++)
                {
                    writer.Write(_conformalCalibrationScores[i]);
                }
            }
            _decisionHead.WriteWeights(writer);
            _noulHead.WriteWeights(writer);
            _scoreHead.WriteWeights(writer);
        }
        finally
        {
            _rwLock.ExitReadLock();
        }
    }

    /// <summary>
    /// Restores all three calibrated head weights and conformal parameters from a binary checkpoint stream.
    /// </summary>
    public void LoadCheckpoint(Stream stream)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(stream);

        _rwLock.EnterWriteLock();
        try
        {
            using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
            uint magic = reader.ReadUInt32();
            if (magic != 0x52564C43u)
                throw new InvalidDataException("Invalid ClavierSession checkpoint magic header.");
            int ver = reader.ReadInt32();
            if (ver != 1 && ver != 2)
                throw new InvalidDataException($"Unsupported ClavierSession checkpoint version {ver}.");
            int embDim = reader.ReadInt32();
            int hidDim = reader.ReadInt32();
            int actions = reader.ReadInt32();
            if (embDim != _embeddingDim || hidDim != _hiddenDim || actions != _numActions)
                throw new InvalidDataException($"Checkpoint dimensions ({embDim},{hidDim},{actions}) do not match session ({_embeddingDim},{_hiddenDim},{_numActions}).");

            if (ver == 1)
            {
                _conformalQuantile = 0.95f;
                _conformalAlpha = 0.05f;
                _conformalCalibrationScores = null;
            }
            else if (ver == 2)
            {
                _conformalQuantile = reader.ReadSingle();
                _conformalAlpha = reader.ReadSingle();
                int calCount = reader.ReadInt32();
                if (calCount > 0)
                {
                    _conformalCalibrationScores = new float[calCount];
                    for (int i = 0; i < calCount; i++)
                    {
                        _conformalCalibrationScores[i] = reader.ReadSingle();
                    }
                }
                else
                {
                    _conformalCalibrationScores = null;
                }
            }

            _decisionHead.ReadWeights(reader);
            _noulHead.ReadWeights(reader);
            _scoreHead.ReadWeights(reader);
        }
        finally
        {
            _rwLock.ExitWriteLock();
        }
    }

    private void ExtractCombinedEmbedding(ReadOnlySpan<char> stateText, ReadOnlySpan<char> secondaryText, Span<float> destination)
    {
        if (secondaryText.IsEmpty)
        {
            ExtractStateEmbedding(stateText, destination);
            return;
        }

        if (_encoder != null)
        {
            int totalLen = stateText.Length + secondaryText.Length + 7;
            char[]? rented = null;
            Span<char> combined = totalLen <= 512
                ? stackalloc char[totalLen]
                : (rented = ArrayPool<char>.Shared.Rent(totalLen)).AsSpan(0, totalLen);

            try
            {
                stateText.CopyTo(combined);
                " [SEP] ".AsSpan().CopyTo(combined.Slice(stateText.Length));
                secondaryText.CopyTo(combined.Slice(stateText.Length + 7));
                _encoder(combined, destination);
            }
            finally
            {
                if (rented != null)
                {
                    ArrayPool<char>.Shared.Return(rented);
                }
            }
            return;
        }

        // Partitioned interaction hashing when using default feature hashing:
        // first half encodes state features, second half encodes secondary/candidate features + cross-interaction!
        destination.Clear();
        int half = destination.Length / 2;
        if (half >= 4)
        {
            AccumulateHashedNgrams(stateText, destination.Slice(0, half), seed: 2166136261u);
            AccumulateHashedNgrams(secondaryText, destination.Slice(half), seed: 2654435761u);

            // Cross-term interaction features so specific (state, action) pairs are linearly & non-linearly separable
            uint stateSummary = HashSpan(stateText);
            AccumulateHashedNgrams(secondaryText, destination, seed: stateSummary ^ 0x9E3779B9u);
            NormalizeL2(destination);
        }
        else
        {
            AccumulateHashedNgrams(stateText, destination, seed: 2166136261u);
            AccumulateHashedNgrams(secondaryText, destination, seed: 2654435761u);
            NormalizeL2(destination);
        }
    }

    private void ExtractStateEmbedding(ReadOnlySpan<char> text, Span<float> destination)
    {
        if (_encoder != null)
        {
            _encoder(text, destination);
        }
        else
        {
            ComputeFallbackEmbedding(text, destination);
        }
    }

    private static void ComputeFallbackEmbedding(ReadOnlySpan<char> text, Span<float> dst)
    {
        dst.Clear();
        if (text.IsEmpty) return;

        AccumulateHashedNgrams(text, dst, seed: 2166136261u);
        NormalizeL2(dst);
    }

    private static uint HashSpan(ReadOnlySpan<char> text)
    {
        uint h = 2166136261u;
        for (int i = 0; i < text.Length; i++)
        {
            h = (h ^ char.ToLowerInvariant(text[i])) * 16777619u;
        }
        return h;
    }

    private static void AccumulateHashedNgrams(ReadOnlySpan<char> text, Span<float> dst, uint seed)
    {
        if (text.IsEmpty || dst.IsEmpty) return;

        uint dim = (uint)dst.Length;
        for (int i = 0; i < text.Length; i++)
        {
            uint c0 = char.ToLowerInvariant(text[i]);
            uint h1 = (seed ^ c0) * 16777619u;
            dst[(int)(h1 % dim)] += (h1 & 0x80000000u) == 0 ? 1.0f : -1.0f;

            if (i + 1 < text.Length)
            {
                uint c1 = char.ToLowerInvariant(text[i + 1]);
                uint h2 = (h1 ^ c1) * 16777619u;
                dst[(int)(h2 % dim)] += (h2 & 0x80000000u) == 0 ? 1.25f : -1.25f;

                if (i + 2 < text.Length)
                {
                    uint c2 = char.ToLowerInvariant(text[i + 2]);
                    uint h3 = (h2 ^ c2) * 16777619u;
                    dst[(int)(h3 % dim)] += (h3 & 0x80000000u) == 0 ? 1.5f : -1.5f;
                }
            }
        }
    }

    private static void NormalizeL2(Span<float> dst)
    {
        float sumSq = SimdKernels.SumOfSquares(dst);
        if (sumSq > 0f)
        {
            float inv = 1.0f / MathF.Sqrt(sumSq);
            SimdKernels.Multiply(dst, inv);
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _decisionHead.Dispose();
            _rwLock.Dispose();
        }
    }
}
