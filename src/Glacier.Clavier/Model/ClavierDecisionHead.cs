namespace Glacier.Clavier.Model;

using System;
using System.Buffers;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

/// <summary>
/// Calibrated multi-layer perceptron decision head.
/// Projects pooled environment state representations (D) -> Hidden (H, ReLU) -> K Action Logits or dynamic candidate compatibility scores with temperature scaling, online cross-entropy training, and binary checkpoint persistence.
/// </summary>
public sealed class ClavierDecisionHead : IDisposable
{
    private readonly int _inputDim;
    private readonly int _hiddenDim;
    private readonly int _numActions;
    private float _temperature;

    // Layer 1: [hiddenDim, inputDim] + bias [hiddenDim]
    private float[] _w1;
    private float[] _b1;

    // Layer 2 (fixed action slots): [numActions, hiddenDim] + bias [numActions]
    private float[] _w2;
    private float[] _b2;

    // Dynamic text candidate compatibility projection: [hiddenDim] + bias
    private float[] _wCand;
    private float _bCand;

    public int InputDim => _inputDim;
    public int HiddenDim => _hiddenDim;
    public int NumActions => _numActions;
    public float Temperature
    {
        get => _temperature;
        set => _temperature = float.IsFinite(value) && value > 0f
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), "Temperature must be finite and positive.");
    }

    public ClavierDecisionHead(int inputDim = 768, int hiddenDim = 256, int numActions = 64, float temperature = 1.0f)
    {
        if (inputDim <= 0 || hiddenDim <= 0 || numActions <= 0)
            throw new ArgumentException("Dimensions must be positive.");

        _inputDim = inputDim;
        _hiddenDim = hiddenDim;
        _numActions = numActions;
        _temperature = float.IsFinite(temperature) && temperature > 0f ? temperature : 1.0f;

        _w1 = new float[checked(hiddenDim * inputDim)];
        _b1 = new float[hiddenDim];
        _w2 = new float[checked(numActions * hiddenDim)];
        _b2 = new float[numActions];
        _wCand = new float[hiddenDim];
        _bCand = 0f;

        InitializeHeWeights();
    }

    private void InitializeHeWeights()
    {
        var rand = new Random(42);
        float std1 = MathF.Sqrt(2.0f / _inputDim);
        for (int i = 0; i < _w1.Length; i++)
        {
            _w1[i] = (float)(rand.NextDouble() * 2.0 - 1.0) * std1;
        }

        for (int h = 0; h < _b1.Length; h++)
        {
            _b1[h] = 0.01f; // Small positive ReLU bias to prevent dead neurons
        }

        float std2 = MathF.Sqrt(2.0f / _hiddenDim);
        for (int i = 0; i < _w2.Length; i++)
        {
            _w2[i] = (float)(rand.NextDouble() * 2.0 - 1.0) * std2;
        }

        for (int h = 0; h < _wCand.Length; h++)
        {
            _wCand[h] = (float)(rand.NextDouble() * 2.0 - 1.0) * std2;
        }
    }

    /// <summary>
    /// Evaluates a single dynamic candidate's compatibility logit from a joint (state, candidateAction) embedding.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public float ForwardCandidateLogit(ReadOnlySpan<float> jointEmbedding, Span<float> hiddenBuffer)
    {
        if (jointEmbedding.Length < _inputDim)
            throw new ArgumentException($"Joint embedding length {jointEmbedding.Length} is less than input dimension {_inputDim}.");
        if (hiddenBuffer.Length < _hiddenDim)
            throw new ArgumentException($"Hidden buffer length {hiddenBuffer.Length} is less than hidden dimension {_hiddenDim}.");

        var stateSlice = jointEmbedding.Slice(0, _inputDim);
        for (int h = 0; h < _hiddenDim; h++)
        {
            ReadOnlySpan<float> wRow = _w1.AsSpan(h * _inputDim, _inputDim);
            float acc = SimdKernels.Dot(stateSlice, wRow) + _b1[h];
            hiddenBuffer[h] = MathF.Max(0.0f, acc);
        }

        float raw = SimdKernels.Dot(hiddenBuffer.Slice(0, _hiddenDim), _wCand.AsSpan(0, _hiddenDim)) + _bCand;
        return raw / _temperature;
    }

    /// <summary>
    /// Evaluates the fixed-slot decision head forward pass with zero heap allocations into caller-provided spans.
    /// Computes calibrated softmax probabilities and emits top decision.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public ClavierDecision Forward(
        ReadOnlySpan<float> stateEmbedding,
        Span<float> hiddenBuffer,
        Span<float> logitsBuffer,
        Span<float> probabilitiesOut,
        int activeActions = -1)
    {
        int actions = (activeActions > 0 && activeActions <= _numActions) ? activeActions : _numActions;
        if (stateEmbedding.Length < _inputDim)
            throw new ArgumentException($"State embedding length {stateEmbedding.Length} is less than input dimension {_inputDim}.");
        if (hiddenBuffer.Length < _hiddenDim)
            throw new ArgumentException($"Hidden buffer length {hiddenBuffer.Length} is less than hidden dimension {_hiddenDim}.");
        if (logitsBuffer.Length < actions)
            throw new ArgumentException($"Logits buffer length {logitsBuffer.Length} is less than actions count {actions}.");

        var stateSlice = stateEmbedding.Slice(0, _inputDim);
        for (int h = 0; h < _hiddenDim; h++)
        {
            ReadOnlySpan<float> wRow = _w1.AsSpan(h * _inputDim, _inputDim);
            float acc = SimdKernels.Dot(stateSlice, wRow) + _b1[h];
            hiddenBuffer[h] = MathF.Max(0.0f, acc);
        }

        float invT = 1.0f / _temperature;
        float maxLogit = float.NegativeInfinity;
        var hiddenSlice = hiddenBuffer.Slice(0, _hiddenDim);
        for (int a = 0; a < actions; a++)
        {
            ReadOnlySpan<float> wRow = _w2.AsSpan(a * _hiddenDim, _hiddenDim);
            float scaled = (SimdKernels.Dot(hiddenSlice, wRow) + _b2[a]) * invT;
            logitsBuffer[a] = scaled;
            if (scaled > maxLogit)
            {
                maxLogit = scaled;
            }
        }

        return SoftmaxAndArgmax(logitsBuffer.Slice(0, actions), probabilitiesOut, maxLogit);
    }

    internal static ClavierDecision SoftmaxAndArgmax(Span<float> scaledLogits, Span<float> probabilitiesOut, float maxLogit)
    {
        int actions = scaledLogits.Length;
        if (actions == 0)
        {
            return new ClavierDecision(0, 0f);
        }

        bool validInputs = float.IsFinite(maxLogit);
        if (validInputs)
        {
            for (int a = 0; a < actions; a++)
            {
                if (!float.IsFinite(scaledLogits[a]))
                {
                    validInputs = false;
                    break;
                }
            }
        }

        if (!validInputs)
        {
            float uniform = 1.0f / actions;
            if (!probabilitiesOut.IsEmpty)
            {
                int copyLen = Math.Min(actions, probabilitiesOut.Length);
                probabilitiesOut.Slice(0, copyLen).Fill(uniform);
            }
            return new ClavierDecision(0, uniform);
        }

        float sumExp = 0f;
        for (int a = 0; a < actions; a++)
        {
            float expVal = MathF.Exp(scaledLogits[a] - maxLogit);
            scaledLogits[a] = expVal;
            sumExp += expVal;
        }

        if (sumExp <= 0f || !float.IsFinite(sumExp))
        {
            float uniform = 1.0f / actions;
            if (!probabilitiesOut.IsEmpty)
            {
                int copyLen = Math.Min(actions, probabilitiesOut.Length);
                probabilitiesOut.Slice(0, copyLen).Fill(uniform);
            }
            return new ClavierDecision(0, uniform);
        }

        float invSum = 1.0f / sumExp;
        uint bestAction = 0;
        float bestProb = -1f;

        for (int a = 0; a < actions; a++)
        {
            float prob = scaledLogits[a] * invSum;
            if (!probabilitiesOut.IsEmpty && a < probabilitiesOut.Length)
            {
                probabilitiesOut[a] = prob;
            }
            if (prob > bestProb)
            {
                bestProb = prob;
                bestAction = (uint)a;
            }
        }

        return new ClavierDecision(bestAction, bestProb);
    }

    /// <summary>
    /// Performs a single supervised cross-entropy gradient step on fixed action slots.
    /// Returns the pre-update categorical cross-entropy loss.
    /// </summary>
    public float TrainStep(
        ReadOnlySpan<float> stateEmbedding,
        int targetAction,
        float learningRate = 0.05f,
        float labelSmoothing = 0.0f)
    {
        if (stateEmbedding.Length < _inputDim)
            throw new ArgumentException($"State embedding length {stateEmbedding.Length} < {_inputDim}.");
        if ((uint)targetAction >= (uint)_numActions)
            throw new ArgumentOutOfRangeException(nameof(targetAction));
        if (!float.IsFinite(learningRate) || learningRate <= 0f)
            throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be finite and positive.");
        if (!float.IsFinite(labelSmoothing) || labelSmoothing < 0f || labelSmoothing > 1f)
            throw new ArgumentOutOfRangeException(nameof(labelSmoothing), "Label smoothing must be finite and in [0, 1].");

        var stateSlice = stateEmbedding.Slice(0, _inputDim);
        EnsureFinite(stateSlice, nameof(stateEmbedding));

        float[]? rentedHidden = null;
        float[]? rentedProbs = null;
        float[]? rentedDHidden = null;

        Span<float> hidden = _hiddenDim <= 512 ? stackalloc float[_hiddenDim] : (rentedHidden = ArrayPool<float>.Shared.Rent(_hiddenDim)).AsSpan(0, _hiddenDim);
        Span<float> probs = _numActions <= 256 ? stackalloc float[_numActions] : (rentedProbs = ArrayPool<float>.Shared.Rent(_numActions)).AsSpan(0, _numActions);
        Span<float> dHidden = _hiddenDim <= 512 ? stackalloc float[_hiddenDim] : (rentedDHidden = ArrayPool<float>.Shared.Rent(_hiddenDim)).AsSpan(0, _hiddenDim);

        try
        {
            for (int h = 0; h < _hiddenDim; h++)
            {
                float pre = SimdKernels.Dot(stateSlice, _w1.AsSpan(h * _inputDim, _inputDim)) + _b1[h];
                hidden[h] = MathF.Max(0f, pre);
            }

            float invT = 1.0f / _temperature;
            float maxLogit = float.NegativeInfinity;
            for (int a = 0; a < _numActions; a++)
            {
                float s = (SimdKernels.Dot(hidden, _w2.AsSpan(a * _hiddenDim, _hiddenDim)) + _b2[a]) * invT;
                probs[a] = s;
                if (s > maxLogit) maxLogit = s;
            }

            float sumExp = 0f;
            for (int a = 0; a < _numActions; a++)
            {
                float e = MathF.Exp(probs[a] - maxLogit);
                probs[a] = e;
                sumExp += e;
            }

            float invSum = 1.0f / MathF.Max(sumExp, 1e-12f);
            float smooth = labelSmoothing / _numActions;
            float loss = 0f;
            dHidden.Clear();

            for (int a = 0; a < _numActions; a++)
            {
                float p = probs[a] * invSum;
                float y = (a == targetAction ? 1.0f - labelSmoothing : 0.0f) + smooth;
                loss -= y * MathF.Log(MathF.Max(p, 1e-9f));

                float dLogit = (p - y) * invT;
                Span<float> w2Row = _w2.AsSpan(a * _hiddenDim, _hiddenDim);
                for (int h = 0; h < _hiddenDim; h++)
                {
                    dHidden[h] += dLogit * w2Row[h];
                    w2Row[h] -= learningRate * dLogit * hidden[h];
                }
                _b2[a] -= learningRate * dLogit;
            }

            for (int h = 0; h < _hiddenDim; h++)
            {
                if (hidden[h] <= 0f) continue;
                float dh = dHidden[h];
                Span<float> w1Row = _w1.AsSpan(h * _inputDim, _inputDim);
                for (int d = 0; d < _inputDim; d++)
                {
                    w1Row[d] -= learningRate * dh * stateSlice[d];
                }
                _b1[h] -= learningRate * dh;
            }

            return loss;
        }
        finally
        {
            if (rentedHidden != null) ArrayPool<float>.Shared.Return(rentedHidden);
            if (rentedProbs != null) ArrayPool<float>.Shared.Return(rentedProbs);
            if (rentedDHidden != null) ArrayPool<float>.Shared.Return(rentedDHidden);
        }
    }

    /// <summary>
    /// Performs a supervised cross-entropy gradient step over dynamic candidate joint embeddings [candidateCount * inputDim].
    /// </summary>
    public float TrainCandidateStep(
        ReadOnlySpan<float> candidateJointEmbeddings,
        int candidateCount,
        int targetCandidateIndex,
        float learningRate = 0.05f,
        float labelSmoothing = 0.0f)
    {
        if (candidateCount <= 0) throw new ArgumentOutOfRangeException(nameof(candidateCount));
        if ((uint)targetCandidateIndex >= (uint)candidateCount) throw new ArgumentOutOfRangeException(nameof(targetCandidateIndex));
        if (!float.IsFinite(learningRate) || learningRate <= 0f)
            throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be finite and positive.");
        if (!float.IsFinite(labelSmoothing) || labelSmoothing < 0f || labelSmoothing > 1f)
            throw new ArgumentOutOfRangeException(nameof(labelSmoothing), "Label smoothing must be finite and in [0, 1].");

        int requiredLen = checked(candidateCount * _inputDim);
        if (candidateJointEmbeddings.Length < requiredLen)
            throw new ArgumentException("Insufficient joint embeddings buffer length.");
        EnsureFinite(candidateJointEmbeddings.Slice(0, requiredLen), nameof(candidateJointEmbeddings));

        int totalHiddenLen = checked(candidateCount * _hiddenDim);
        float[] hiddens = ArrayPool<float>.Shared.Rent(totalHiddenLen);
        float[] probs = ArrayPool<float>.Shared.Rent(candidateCount);
        float[]? rentedDWCand = null;
        try
        {
            float maxLogit = float.NegativeInfinity;
            for (int c = 0; c < candidateCount; c++)
            {
                var emb = candidateJointEmbeddings.Slice(c * _inputDim, _inputDim);
                var hid = hiddens.AsSpan(c * _hiddenDim, _hiddenDim);
                float scaled = ForwardCandidateLogit(emb, hid);
                probs[c] = scaled;
                if (scaled > maxLogit) maxLogit = scaled;
            }

            float sumExp = 0f;
            for (int c = 0; c < candidateCount; c++)
            {
                float e = MathF.Exp(probs[c] - maxLogit);
                probs[c] = e;
                sumExp += e;
            }

            float invSum = 1.0f / MathF.Max(sumExp, 1e-12f);
            float invT = 1.0f / _temperature;
            float smooth = labelSmoothing / candidateCount;
            float loss = 0f;

            Span<float> dWCand = _hiddenDim <= 512
                ? stackalloc float[_hiddenDim]
                : (rentedDWCand = ArrayPool<float>.Shared.Rent(_hiddenDim)).AsSpan(0, _hiddenDim);
            dWCand.Clear();
            float dBCand = 0f;

            for (int c = 0; c < candidateCount; c++)
            {
                float p = probs[c] * invSum;
                float y = (c == targetCandidateIndex ? 1.0f - labelSmoothing : 0.0f) + smooth;
                loss -= y * MathF.Log(MathF.Max(p, 1e-9f));
                float dLogit = (p - y) * invT;

                var emb = candidateJointEmbeddings.Slice(c * _inputDim, _inputDim);
                var hid = hiddens.AsSpan(c * _hiddenDim, _hiddenDim);

                dBCand += dLogit;
                for (int h = 0; h < _hiddenDim; h++)
                {
                    dWCand[h] += dLogit * hid[h];
                    if (hid[h] > 0f)
                    {
                        float dh = dLogit * _wCand[h];
                        Span<float> w1Row = _w1.AsSpan(h * _inputDim, _inputDim);
                        for (int d = 0; d < _inputDim; d++)
                        {
                            w1Row[d] -= learningRate * dh * emb[d];
                        }
                        _b1[h] -= learningRate * dh;
                    }
                }
            }

            for (int h = 0; h < _hiddenDim; h++)
            {
                _wCand[h] -= learningRate * dWCand[h];
            }
            _bCand -= learningRate * dBCand;

            return loss;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(hiddens);
            ArrayPool<float>.Shared.Return(probs);
            if (rentedDWCand != null) ArrayPool<float>.Shared.Return(rentedDWCand);
        }
    }

    public void SaveWeights(Stream stream)
    {
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        WriteWeights(writer);
    }

    public void LoadWeights(Stream stream)
    {
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        ReadWeights(reader);
    }

    internal void WriteWeights(BinaryWriter writer)
    {
        writer.Write(0x48444C43u); // "CLDH"
        writer.Write(_inputDim);
        writer.Write(_hiddenDim);
        writer.Write(_numActions);
        writer.Write(_temperature);
        WriteFloatArray(writer, _w1);
        WriteFloatArray(writer, _b1);
        WriteFloatArray(writer, _w2);
        WriteFloatArray(writer, _b2);
        WriteFloatArray(writer, _wCand);
        writer.Write(_bCand);
    }

    internal readonly struct StagedWeights
    {
        public readonly float Temperature;
        public readonly float[] W1;
        public readonly float[] B1;
        public readonly float[] W2;
        public readonly float[] B2;
        public readonly float[] WCand;
        public readonly float BCand;

        public StagedWeights(
            float temperature,
            float[] w1,
            float[] b1,
            float[] w2,
            float[] b2,
            float[] wCand,
            float bCand)
        {
            Temperature = temperature;
            W1 = w1;
            B1 = b1;
            W2 = w2;
            B2 = b2;
            WCand = wCand;
            BCand = bCand;
        }
    }

    internal StagedWeights StageReadWeights(BinaryReader reader)
    {
        uint magic = reader.ReadUInt32();
        if (magic != 0x48444C43u)
            throw new InvalidDataException("Invalid ClavierDecisionHead magic header.");
        int inDim = reader.ReadInt32();
        int hidDim = reader.ReadInt32();
        int actions = reader.ReadInt32();
        if (inDim != _inputDim || hidDim != _hiddenDim || actions != _numActions)
            throw new InvalidDataException($"DecisionHead shape mismatch: expected ({_inputDim},{_hiddenDim},{_numActions}), got ({inDim},{hidDim},{actions}).");
        float temp = reader.ReadSingle();
        if (!float.IsFinite(temp) || temp <= 0f)
            throw new InvalidDataException($"Corrupted ClavierDecisionHead checkpoint temperature: {temp}.");

        float[] w1 = new float[_w1.Length];
        float[] b1 = new float[_b1.Length];
        float[] w2 = new float[_w2.Length];
        float[] b2 = new float[_b2.Length];
        float[] wCand = new float[_wCand.Length];

        ReadFloatArray(reader, w1);
        ReadFloatArray(reader, b1);
        ReadFloatArray(reader, w2);
        ReadFloatArray(reader, b2);
        ReadFloatArray(reader, wCand);

        float bCand = reader.ReadSingle();
        if (!float.IsFinite(bCand))
            throw new InvalidDataException($"Corrupted ClavierDecisionHead checkpoint candidate bias: {bCand}.");

        return new StagedWeights(temp, w1, b1, w2, b2, wCand, bCand);
    }

    internal void CommitWeights(in StagedWeights staged)
    {
        _w1 = staged.W1;
        _b1 = staged.B1;
        _w2 = staged.W2;
        _b2 = staged.B2;
        _wCand = staged.WCand;
        _temperature = staged.Temperature;
        _bCand = staged.BCand;
    }

    internal void ReadWeights(BinaryReader reader)
    {
        StagedWeights staged = StageReadWeights(reader);
        CommitWeights(in staged);
    }

    internal static void WriteFloatArray(BinaryWriter writer, float[] values)
    {
        if (values.Length == 0) return;

        if (BitConverter.IsLittleEndian)
        {
            writer.Write(MemoryMarshal.AsBytes(values.AsSpan()));
            return;
        }

        for (int i = 0; i < values.Length; i++)
        {
            writer.Write(values[i]);
        }
    }

    internal static void ReadFloatArray(BinaryReader reader, float[] values)
    {
        if (values.Length == 0) return;

        if (BitConverter.IsLittleEndian)
        {
            Span<byte> dest = MemoryMarshal.AsBytes(values.AsSpan());
            while (!dest.IsEmpty)
            {
                int read = reader.Read(dest);
                if (read == 0)
                    throw new EndOfStreamException("Unexpected end of stream while reading weight array.");
                dest = dest.Slice(read);
            }
        }
        else
        {
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = reader.ReadSingle();
            }
        }

        for (int i = 0; i < values.Length; i++)
        {
            if (!float.IsFinite(values[i]))
                throw new InvalidDataException($"Corrupted checkpoint contains non-finite float value at index {i}.");
        }
    }

    internal static void EnsureFinite(ReadOnlySpan<float> values, string paramName)
    {
        for (int i = 0; i < values.Length; i++)
        {
            if (!float.IsFinite(values[i]))
                throw new ArgumentException($"Input contains non-finite value at index {i}.", paramName);
        }
    }

    internal static float Log1p(float x)
    {
        if (x == 0.0f) return 0.0f;
        double xd = x;
        double u = 1.0 + xd;
        if (u == 1.0) return x;
        return (float)(Math.Log(u) * (xd / (u - 1.0)));
    }

    public void Dispose() { }
}
