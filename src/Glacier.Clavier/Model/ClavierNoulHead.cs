namespace Glacier.Clavier.Model;

using System;
using System.Buffers;
using System.IO;
using System.Runtime.CompilerServices;
using System.Text;

/// <summary>
/// Non-autoregressive binary decision and verification head (Jev 'Noul' primitive).
/// Maps contextual state embeddings to calibrated probability p in [0, 1] via temperature-scaled sigmoid, with online binary cross-entropy training and weight persistence.
/// </summary>
public sealed class ClavierNoulHead
{
    private readonly int _inputDim;
    private readonly int _hiddenDim;
    private float _temperature;

    private readonly float[] _w1; // [hiddenDim, inputDim]
    private readonly float[] _b1; // [hiddenDim]
    private readonly float[] _w2; // [hiddenDim]
    private float _b2;

    public int InputDim => _inputDim;
    public int HiddenDim => _hiddenDim;
    public float Temperature
    {
        get => _temperature;
        set => _temperature = MathF.Max(0.01f, value);
    }

    public ClavierNoulHead(int inputDim = 768, int hiddenDim = 256, float temperature = 1.0f)
    {
        if (inputDim <= 0 || hiddenDim <= 0)
            throw new ArgumentException("Dimensions must be positive.");

        _inputDim = inputDim;
        _hiddenDim = hiddenDim;
        _temperature = MathF.Max(0.01f, temperature);

        _w1 = new float[hiddenDim * inputDim];
        _b1 = new float[hiddenDim];
        _w2 = new float[hiddenDim];
        _b2 = 0f;

        InitializeWeights();
    }

    private void InitializeWeights()
    {
        float scale1 = MathF.Sqrt(2.0f / _inputDim);
        var rng = new Random(1337);
        for (int i = 0; i < _w1.Length; i++)
        {
            _w1[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * scale1;
        }

        for (int h = 0; h < _b1.Length; h++)
        {
            _b1[h] = 0.01f;
        }

        float scale2 = MathF.Sqrt(2.0f / _hiddenDim);
        for (int i = 0; i < _w2.Length; i++)
        {
            _w2[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * scale2;
        }
    }

    /// <summary>
    /// Evaluates the Noul binary head forward pass with zero heap allocations.
    /// Emits an 8-byte ClavierNoul containing affirmative decision and calibrated probability.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public ClavierNoul Forward(
        ReadOnlySpan<float> stateEmbedding,
        Span<float> hiddenBuffer,
        float threshold = 0.5f)
    {
        if (stateEmbedding.Length < _inputDim)
            throw new ArgumentException($"State embedding length {stateEmbedding.Length} is less than input dimension {_inputDim}.");
        if (hiddenBuffer.Length < _hiddenDim)
            throw new ArgumentException($"Hidden buffer length {hiddenBuffer.Length} is less than hidden dimension {_hiddenDim}.");

        var stateSlice = stateEmbedding.Slice(0, _inputDim);
        for (int h = 0; h < _hiddenDim; h++)
        {
            ReadOnlySpan<float> wRow = _w1.AsSpan(h * _inputDim, _inputDim);
            float acc = SimdKernels.Dot(stateSlice, wRow) + _b1[h];
            hiddenBuffer[h] = MathF.Max(0.0f, acc);
        }

        float dot2 = SimdKernels.Dot(hiddenBuffer.Slice(0, _hiddenDim), _w2.AsSpan(0, _hiddenDim));
        float logit = (dot2 + _b2) / _temperature;

        float prob = 1.0f / (1.0f + MathF.Exp(-logit));
        bool isAffirmative = prob >= threshold;

        return new ClavierNoul(isAffirmative, prob);
    }

    /// <summary>
    /// Performs a single binary cross-entropy gradient update and returns pre-update BCE loss.
    /// </summary>
    public float TrainStep(ReadOnlySpan<float> stateEmbedding, bool targetAffirmative, float learningRate = 0.05f)
    {
        if (stateEmbedding.Length < _inputDim)
            throw new ArgumentException($"State embedding length {stateEmbedding.Length} < {_inputDim}.");
        if (learningRate <= 0f)
            throw new ArgumentOutOfRangeException(nameof(learningRate));

        float[]? rented = null;
        Span<float> hidden = _hiddenDim <= 512 ? stackalloc float[_hiddenDim] : (rented = ArrayPool<float>.Shared.Rent(_hiddenDim)).AsSpan(0, _hiddenDim);
        try
        {
            var stateSlice = stateEmbedding.Slice(0, _inputDim);
            for (int h = 0; h < _hiddenDim; h++)
            {
                float acc = SimdKernels.Dot(stateSlice, _w1.AsSpan(h * _inputDim, _inputDim)) + _b1[h];
                hidden[h] = MathF.Max(0.0f, acc);
            }

            float invT = 1.0f / _temperature;
            float logit = (SimdKernels.Dot(hidden, _w2) + _b2) * invT;
            float prob = 1.0f / (1.0f + MathF.Exp(-logit));
            float y = targetAffirmative ? 1.0f : 0.0f;

            float loss = -(y * MathF.Log(MathF.Max(prob, 1e-7f)) + (1.0f - y) * MathF.Log(MathF.Max(1.0f - prob, 1e-7f)));
            float dLogit = (prob - y) * invT;

            for (int h = 0; h < _hiddenDim; h++)
            {
                float w2Old = _w2[h];
                _w2[h] -= learningRate * dLogit * hidden[h];
                if (hidden[h] > 0f)
                {
                    float dh = dLogit * w2Old;
                    Span<float> w1Row = _w1.AsSpan(h * _inputDim, _inputDim);
                    for (int d = 0; d < _inputDim; d++)
                    {
                        w1Row[d] -= learningRate * dh * stateSlice[d];
                    }
                    _b1[h] -= learningRate * dh;
                }
            }
            _b2 -= learningRate * dLogit;

            return loss;
        }
        finally
        {
            if (rented != null) ArrayPool<float>.Shared.Return(rented);
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
        writer.Write(0x484E4C43u); // "CLNH"
        writer.Write(_inputDim);
        writer.Write(_hiddenDim);
        writer.Write(_temperature);
        ClavierDecisionHead.WriteFloatArray(writer, _w1);
        ClavierDecisionHead.WriteFloatArray(writer, _b1);
        ClavierDecisionHead.WriteFloatArray(writer, _w2);
        writer.Write(_b2);
    }

    internal void ReadWeights(BinaryReader reader)
    {
        uint magic = reader.ReadUInt32();
        if (magic != 0x484E4C43u)
            throw new InvalidDataException("Invalid ClavierNoulHead magic header.");
        int inDim = reader.ReadInt32();
        int hidDim = reader.ReadInt32();
        if (inDim != _inputDim || hidDim != _hiddenDim)
            throw new InvalidDataException($"NoulHead shape mismatch: expected ({_inputDim},{_hiddenDim}), got ({inDim},{hidDim}).");
        _temperature = reader.ReadSingle();
        ClavierDecisionHead.ReadFloatArray(reader, _w1);
        ClavierDecisionHead.ReadFloatArray(reader, _b1);
        ClavierDecisionHead.ReadFloatArray(reader, _w2);
        _b2 = reader.ReadSingle();
    }
}
