namespace Glacier.Clavier.Engine;

using System;
using System.Diagnostics;
using Glacier.Clavier.Model;

/// <summary>
/// Delegate for extracting state representation vectors into caller-provided float spans.
/// Compatible with Glacier.Inference, ONNX, and custom embedding models.
/// </summary>
public delegate void ClavierStateEncoder(ReadOnlySpan<char> text, Span<float> destination);

/// <summary>
/// High-throughput execution session for sub-millisecond System-1 discrete policy inference.
/// Coordinates unmasked bidirectional representation, state pooling, and calibrated action head.
/// </summary>
public sealed class ClavierSession : IDisposable
{
    private readonly ClavierDecisionHead _decisionHead;
    private readonly ClavierNoulHead _noulHead;
    private readonly ClavierScoreHead _scoreHead;
    private readonly ClavierStateEncoder? _encoder;
    private readonly int _embeddingDim;
    private readonly int _numActions;
    private readonly float[] _stateEmbBuffer;
    private readonly float[] _hiddenBuffer;
    private readonly float[] _logitsBuffer;
    private readonly float[] _probsBuffer;
    private readonly object _syncLock = new();
    private bool _disposed;

    public int EmbeddingDim => _embeddingDim;
    public int NumActions => _numActions;
    public float Temperature
    {
        get => _decisionHead.Temperature;
        set
        {
            _decisionHead.Temperature = value;
            _noulHead.Temperature = value;
            _scoreHead.Temperature = value;
        }
    }

    public ClavierSession(
        int embeddingDim = 768,
        int hiddenDim = 256,
        int numActions = 64,
        float temperature = 1.0f,
        ClavierStateEncoder? encoder = null)
    {
        _embeddingDim = embeddingDim;
        _numActions = numActions;
        _encoder = encoder;
        _decisionHead = new ClavierDecisionHead(embeddingDim, hiddenDim, numActions, temperature);
        _noulHead = new ClavierNoulHead(embeddingDim, hiddenDim, temperature);
        _scoreHead = new ClavierScoreHead(embeddingDim, hiddenDim, temperature);

        _stateEmbBuffer = new float[embeddingDim];
        _hiddenBuffer = new float[hiddenDim];
        _logitsBuffer = new float[numActions];
        _probsBuffer = new float[numActions];
    }

    /// <summary>
    /// Evaluates environment state against candidate actions in a single forward pass (&lt; 2.5 ms).
    /// Emits the winning 8-byte ClavierDecision (ActionId, Confidence).
    /// </summary>
    public ClavierDecision Decide(ReadOnlySpan<char> stateText, ReadOnlySpan<string> candidateActions = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_syncLock)
        {
            // 1. Extract bidirectional state embedding
            ExtractStateEmbedding(stateText, _stateEmbBuffer);

            // 2. Forward pass through decision head
            int activeActions = !candidateActions.IsEmpty ? Math.Min(candidateActions.Length, _numActions) : _numActions;
            var decision = _decisionHead.Forward(
                _stateEmbBuffer.AsSpan(0, _embeddingDim),
                _hiddenBuffer,
                _logitsBuffer.AsSpan(0, activeActions),
                _probsBuffer.AsSpan(0, activeActions),
                activeActions);

            return decision;
        }
    }

    /// <summary>
    /// Evaluates environment state and returns full calibrated probability distribution across actions.
    /// </summary>
    public ClavierDecisionResult DecideWithDistribution(ReadOnlySpan<char> stateText, ReadOnlySpan<string> candidateActions = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var sw = Stopwatch.StartNew();
        int activeActions = !candidateActions.IsEmpty ? Math.Min(candidateActions.Length, _numActions) : _numActions;
        var probs = new float[activeActions];

        ClavierDecision decision;
        lock (_syncLock)
        {
            ExtractStateEmbedding(stateText, _stateEmbBuffer);

            decision = _decisionHead.Forward(
                _stateEmbBuffer.AsSpan(0, _embeddingDim),
                _hiddenBuffer,
                _logitsBuffer.AsSpan(0, activeActions),
                probs,
                activeActions);
        }
        sw.Stop();

        string? actionName = !candidateActions.IsEmpty && decision.ActionId < candidateActions.Length
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

    /// <summary>
    /// Batch-evaluates multiple independent environment states concurrently.
    /// </summary>
    public void DecideBatch(
        ReadOnlySpan<ReadOnlyMemory<char>> states,
        Span<ClavierDecision> decisions,
        ReadOnlySpan<string> candidateActions = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        int count = Math.Min(states.Length, decisions.Length);
        for (int i = 0; i < count; i++)
        {
            decisions[i] = Decide(states[i].Span, candidateActions);
        }
    }

    /// <summary>
    /// Evaluates an assertion or hypothesis against an environment state in a single forward pass (Jev 'Noul' primitive).
    /// Returns a blittable 8-byte ClavierNoul containing affirmative boolean and calibrated probability.
    /// </summary>
    public ClavierNoul Verify(ReadOnlySpan<char> stateText, ReadOnlySpan<char> hypothesis = default, float threshold = 0.5f)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_syncLock)
        {
            ExtractCombinedEmbedding(stateText, hypothesis, _stateEmbBuffer);
            return _noulHead.Forward(_stateEmbBuffer.AsSpan(0, _embeddingDim), _hiddenBuffer, threshold);
        }
    }

    /// <summary>
    /// Evaluates an assertion or hypothesis with latency timing and calibrated confidence metrics.
    /// </summary>
    public ClavierNoulResult VerifyWithDetails(ReadOnlySpan<char> stateText, ReadOnlySpan<char> hypothesis = default, float threshold = 0.5f)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var sw = Stopwatch.StartNew();
        ClavierNoul noul;
        lock (_syncLock)
        {
            ExtractCombinedEmbedding(stateText, hypothesis, _stateEmbBuffer);
            noul = _noulHead.Forward(_stateEmbBuffer.AsSpan(0, _embeddingDim), _hiddenBuffer, threshold);
        }
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
        for (int i = 0; i < count; i++)
        {
            results[i] = Verify(states[i].Span, hypothesis, threshold);
        }
    }

    /// <summary>
    /// Evaluates continuous scalar rating against criteria on scale [min, max] in a single forward pass (Jev 'Score' primitive).
    /// Returns a blittable 8-byte ClavierScore containing scaled value and calibrated confidence.
    /// </summary>
    public ClavierScore Score(
        ReadOnlySpan<char> stateText,
        ReadOnlySpan<char> criteria = default,
        float min = 0.0f,
        float max = 1.0f)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_syncLock)
        {
            ExtractCombinedEmbedding(stateText, criteria, _stateEmbBuffer);
            return _scoreHead.Forward(_stateEmbBuffer.AsSpan(0, _embeddingDim), _hiddenBuffer, min, max);
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
        ClavierScore score;
        lock (_syncLock)
        {
            ExtractCombinedEmbedding(stateText, criteria, _stateEmbBuffer);
            score = _scoreHead.Forward(_stateEmbBuffer.AsSpan(0, _embeddingDim), _hiddenBuffer, min, max);
        }
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
        for (int i = 0; i < count; i++)
        {
            results[i] = Score(states[i].Span, criteria, min, max);
        }
    }

    private void ExtractCombinedEmbedding(ReadOnlySpan<char> stateText, ReadOnlySpan<char> secondaryText, Span<float> destination)
    {
        if (secondaryText.IsEmpty)
        {
            ExtractStateEmbedding(stateText, destination);
            return;
        }

        int totalLen = stateText.Length + secondaryText.Length + 7;
        char[]? rented = null;
        Span<char> combined = totalLen <= 512
            ? stackalloc char[totalLen]
            : (rented = System.Buffers.ArrayPool<char>.Shared.Rent(totalLen)).AsSpan(0, totalLen);

        try
        {
            stateText.CopyTo(combined);
            " [SEP] ".AsSpan().CopyTo(combined.Slice(stateText.Length));
            secondaryText.CopyTo(combined.Slice(stateText.Length + 7));

            ExtractStateEmbedding(combined, destination);
        }
        finally
        {
            if (rented != null)
            {
                System.Buffers.ArrayPool<char>.Shared.Return(rented);
            }
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
            // Fast deterministic lexical SIMD hash embedding fallback when offline / no GGUF attached
            ComputeFallbackEmbedding(text, destination);
        }
    }

    private void ComputeFallbackEmbedding(ReadOnlySpan<char> text, Span<float> dst)
    {
        dst.Clear();
        if (text.IsEmpty) return;

        int dim = dst.Length;
        for (int i = 0; i < text.Length; i++)
        {
            uint h = (uint)(text[i] * 31 + i);
            int idx = (int)(h % (uint)dim);
            dst[idx] += 1.0f;
        }

        // Vectorized L2 Normalize
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
        }
    }
}
