namespace Glacier.Clavier.Model;

using System;
using System.Runtime.CompilerServices;

/// <summary>
/// Non-autoregressive binary decision and verification head (Jev 'Noul' primitive).
/// Maps contextual state embeddings to calibrated probability p in [0, 1] via temperature-scaled sigmoid.
/// Evaluates assertions, guardrails, safety flags, and binary facts in a single forward pass.
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
        // He / Kaiming normal initialization
        float scale1 = MathF.Sqrt(2.0f / _inputDim);
        var rng = new Random(1337);
        for (int i = 0; i < _w1.Length; i++)
        {
            _w1[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * scale1;
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

        // 1. Dense Layer 1: hidden = ReLU(W1^T * state + b1)
        hiddenBuffer.Slice(0, _hiddenDim).Clear();
        var stateSlice = stateEmbedding.Slice(0, _inputDim);
        for (int h = 0; h < _hiddenDim; h++)
        {
            ReadOnlySpan<float> wRow = _w1.AsSpan(h * _inputDim, _inputDim);
            float dot = SimdKernels.Dot(stateSlice, wRow);
            float acc = dot + _b1[h];
            hiddenBuffer[h] = MathF.Max(0.0f, acc); // ReLU
        }

        // 2. Output Projection: logit = (w2^T * hidden + b2) / Temperature
        float dot2 = SimdKernels.Dot(hiddenBuffer.Slice(0, _hiddenDim), _w2.AsSpan(0, _hiddenDim));
        float logit = (dot2 + _b2) / _temperature;

        // 3. Calibrated Sigmoid
        float prob = 1.0f / (1.0f + MathF.Exp(-logit));
        bool isAffirmative = prob >= threshold;

        return new ClavierNoul(isAffirmative, prob);
    }
}
