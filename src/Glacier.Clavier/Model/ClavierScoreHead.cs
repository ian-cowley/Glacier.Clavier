namespace Glacier.Clavier.Model;

using System;
using System.Runtime.CompilerServices;

/// <summary>
/// Non-autoregressive continuous scalar rating and evaluation head (Jev 'Score' primitive).
/// Maps contextual state embeddings to a calibrated continuous score bounded in [min, max]
/// along with an epistemic confidence rating in a single forward pass.
/// Evaluates risk scores, priority ratings, urgency indices, and alignment metrics.
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
        _inputDim = inputDim;
        _hiddenDim = hiddenDim;
        _temperature = MathF.Max(0.01f, temperature);

        _w1 = new float[hiddenDim * inputDim];
        _b1 = new float[hiddenDim];
        _wVal = new float[hiddenDim];
        _wConf = new float[hiddenDim];
        _bVal = 0f;
        _bConf = 1.0f; // Default positive confidence prior

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

        // 1. Dense Layer 1: hidden = ReLU(W1^T * state + b1)
        hiddenBuffer.Slice(0, _hiddenDim).Clear();
        for (int h = 0; h < _hiddenDim; h++)
        {
            float acc = _b1[h];
            int wOffset = h * _inputDim;
            for (int d = 0; d < _inputDim; d++)
            {
                acc += stateEmbedding[d] * _w1[wOffset + d];
            }
            hiddenBuffer[h] = MathF.Max(0.0f, acc); // ReLU
        }

        // 2. Value Projection: normalized in [0, 1] via temperature sigmoid
        float valLogit = _bVal;
        float confLogit = _bConf;
        for (int h = 0; h < _hiddenDim; h++)
        {
            float hVal = hiddenBuffer[h];
            valLogit += hVal * _wVal[h];
            confLogit += hVal * _wConf[h];
        }

        float normVal = 1.0f / (1.0f + MathF.Exp(-valLogit / _temperature));
        float confidence = 1.0f / (1.0f + MathF.Exp(-confLogit));

        // 3. Map to caller's bounded [min, max] scale
        float scaledValue = min + (max - min) * normVal;

        return new ClavierScore(scaledValue, confidence);
    }
}
