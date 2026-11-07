namespace Glacier.Clavier.Model;

using System;
using System.Buffers;
using System.IO;
using System.Runtime.CompilerServices;
using System.Text;

/// <summary>
/// Non-autoregressive continuous scalar rating and evaluation head (Jev 'Score' primitive).
/// Maps contextual state embeddings to a calibrated continuous score bounded in [min, max]
/// along with an epistemic confidence rating in a single forward pass, supporting online regression training and weight persistence.
/// </summary>
public sealed class ClavierScoreHead
{
    private readonly int _inputDim;
    private readonly int _hiddenDim;
    private float _temperature;

    private readonly float[] _w1;    // [hiddenDim, inputDim]
    private readonly float[] _b1;    // [hiddenDim]
    private readonly float[] _wVal;  // [hiddenDim]
    private float _bVal;
    private readonly float[] _wConf; // [hiddenDim]
    private float _bConf;

    public int InputDim => _inputDim;
    public int HiddenDim => _hiddenDim;
    public float Temperature
    {
        get => _temperature;
        set => _temperature = MathF.Max(0.01f, value);
    }

    public ClavierScoreHead(int inputDim = 768, int hiddenDim = 256, float temperature = 1.0f)
    {
        if (inputDim <= 0 || hiddenDim <= 0)
            throw new ArgumentException("Dimensions must be positive.");

        _inputDim = inputDim;
        _hiddenDim = hiddenDim;
        _temperature = MathF.Max(0.01f, temperature);

        _w1 = new float[hiddenDim * inputDim];
        _b1 = new float[hiddenDim];
        _wVal = new float[hiddenDim];
        _wConf = new float[hiddenDim];
        _bVal = 0f;
        _bConf = 1.0f;

        InitializeWeights();
    }

    private void InitializeWeights()
    {
        float scale1 = MathF.Sqrt(2.0f / _inputDim);
        var rng = new Random(4242);
        for (int i = 0; i < _w1.Length; i++)
        {
            _w1[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * scale1;
        }

        for (int h = 0; h < _b1.Length; h++)
        {
            _b1[h] = 0.01f;
        }

        float scale2 = MathF.Sqrt(2.0f / _hiddenDim);
        for (int i = 0; i < _wVal.Length; i++)
        {
            _wVal[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * scale2;
            _wConf[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * scale2;
        }
    }

    /// <summary>
    /// Evaluates the continuous Score head forward pass with zero heap allocations.
    /// Emits an 8-byte ClavierScore containing the scaled value in [min, max] and confidence.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public ClavierScore Forward(
        ReadOnlySpan<float> stateEmbedding,
        Span<float> hiddenBuffer,
        float min = 0.0f,
        float max = 1.0f)
    {
        if (stateEmbedding.Length < _inputDim)
            throw new ArgumentException($"State embedding length {stateEmbedding.Length} is less than input dimension {_inputDim}.");
        if (hiddenBuffer.Length < _hiddenDim)
            throw new ArgumentException($"Hidden buffer length {hiddenBuffer.Length} is less than hidden dimension {_hiddenDim}.");
        if (max < min)
            throw new ArgumentException($"Max ({max}) cannot be less than min ({min}).");

        var stateSlice = stateEmbedding.Slice(0, _inputDim);
        for (int h = 0; h < _hiddenDim; h++)
        {
            ReadOnlySpan<float> wRow = _w1.AsSpan(h * _inputDim, _inputDim);
            float acc = SimdKernels.Dot(stateSlice, wRow) + _b1[h];
            hiddenBuffer[h] = MathF.Max(0.0f, acc);
        }

        var hiddenSlice = hiddenBuffer.Slice(0, _hiddenDim);
        float valLogit = SimdKernels.Dot(hiddenSlice, _wVal.AsSpan(0, _hiddenDim)) + _bVal;
        float confLogit = SimdKernels.Dot(hiddenSlice, _wConf.AsSpan(0, _hiddenDim)) + _bConf;

        float normVal = 1.0f / (1.0f + MathF.Exp(-valLogit / _temperature));
        float confidence = 1.0f / (1.0f + MathF.Exp(-confLogit));

        float scaledValue = min + (max - min) * normVal;
        return new ClavierScore(scaledValue, confidence);
    }

    /// <summary>
    /// Performs a single supervised gradient update toward targetValue in [min, max] and calibrates epistemic confidence.
    /// Returns normalized squared error loss.
    /// </summary>
    public float TrainStep(
        ReadOnlySpan<float> stateEmbedding,
        float targetValue,
        float min = 0.0f,
        float max = 1.0f,
        float learningRate = 0.05f)
    {
        if (stateEmbedding.Length < _inputDim)
            throw new ArgumentException($"State embedding length {stateEmbedding.Length} < {_inputDim}.");
        if (max <= min)
            throw new ArgumentException($"Max ({max}) must be greater than min ({min}).");
        if (learningRate <= 0f)
            throw new ArgumentOutOfRangeException(nameof(learningRate));

        float targetNorm = Math.Clamp((targetValue - min) / (max - min), 1e-4f, 1.0f - 1e-4f);

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
            float valLogit = (SimdKernels.Dot(hidden, _wVal) + _bVal) * invT;
            float confLogit = SimdKernels.Dot(hidden, _wConf) + _bConf;

            float predNorm = 1.0f / (1.0f + MathF.Exp(-valLogit));
            float predConf = 1.0f / (1.0f + MathF.Exp(-confLogit));

            float diff = predNorm - targetNorm;
            float loss = diff * diff;

            // BCE-style logit gradient for smooth bounded regression + self-calibrated confidence target (1 - |error|)
            float dVal = diff * invT;
            float targetConf = Math.Clamp(1.0f - MathF.Abs(diff), 0.05f, 0.99f);
            float dConf = 0.25f * (predConf - targetConf);

            for (int h = 0; h < _hiddenDim; h++)
            {
                float wValOld = _wVal[h];
                float wConfOld = _wConf[h];
                _wVal[h] -= learningRate * dVal * hidden[h];
                _wConf[h] -= learningRate * dConf * hidden[h];

                if (hidden[h] > 0f)
                {
                    float dh = dVal * wValOld + dConf * wConfOld;
                    Span<float> w1Row = _w1.AsSpan(h * _inputDim, _inputDim);
                    for (int d = 0; d < _inputDim; d++)
                    {
                        w1Row[d] -= learningRate * dh * stateSlice[d];
                    }
                    _b1[h] -= learningRate * dh;
                }
            }

            _bVal -= learningRate * dVal;
            _bConf -= learningRate * dConf;

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
        writer.Write(0x48534C43u); // "CLSH"
        writer.Write(_inputDim);
        writer.Write(_hiddenDim);
        writer.Write(_temperature);
        ClavierDecisionHead.WriteFloatArray(writer, _w1);
        ClavierDecisionHead.WriteFloatArray(writer, _b1);
        ClavierDecisionHead.WriteFloatArray(writer, _wVal);
        writer.Write(_bVal);
        ClavierDecisionHead.WriteFloatArray(writer, _wConf);
        writer.Write(_bConf);
    }

    internal void ReadWeights(BinaryReader reader)
    {
        uint magic = reader.ReadUInt32();
        if (magic != 0x48534C43u)
            throw new InvalidDataException("Invalid ClavierScoreHead magic header.");
        int inDim = reader.ReadInt32();
        int hidDim = reader.ReadInt32();
        if (inDim != _inputDim || hidDim != _hiddenDim)
            throw new InvalidDataException($"ScoreHead shape mismatch: expected ({_inputDim},{_hiddenDim}), got ({inDim},{hidDim}).");
        _temperature = reader.ReadSingle();
        ClavierDecisionHead.ReadFloatArray(reader, _w1);
        ClavierDecisionHead.ReadFloatArray(reader, _b1);
        ClavierDecisionHead.ReadFloatArray(reader, _wVal);
        _bVal = reader.ReadSingle();
        ClavierDecisionHead.ReadFloatArray(reader, _wConf);
        _bConf = reader.ReadSingle();
    }
}
