namespace Glacier.Clavier.Model;

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using Glacier.Tensor.Core;

/// <summary>
/// Calibrated multi-layer perceptron decision head.
/// Projects pooled environment state representations (D=768) -> Hidden (H=256, ReLU) -> K Action Logits with temperature scaling.
/// </summary>
public sealed class ClavierDecisionHead : IDisposable
{
    private readonly int _inputDim;
    private readonly int _hiddenDim;
    private readonly int _numActions;
    private float _temperature;

    // Layer 1: [inputDim, hiddenDim] + bias [hiddenDim]
    private readonly float[] _w1;
    private readonly float[] _b1;

    // Layer 2: [hiddenDim, numActions] + bias [numActions]
    private readonly float[] _w2;
    private readonly float[] _b2;

    public int InputDim => _inputDim;
    public int HiddenDim => _hiddenDim;
    public int NumActions => _numActions;
    public float Temperature
    {
        get => _temperature;
        set => _temperature = value > 0f ? value : throw new ArgumentOutOfRangeException(nameof(value), "Temperature must be positive.");
    }

    public ClavierDecisionHead(int inputDim = 768, int hiddenDim = 256, int numActions = 64, float temperature = 1.0f)
    {
        if (inputDim <= 0 || hiddenDim <= 0 || numActions <= 0)
            throw new ArgumentException("Dimensions must be positive.");

        _inputDim = inputDim;
        _hiddenDim = hiddenDim;
        _numActions = numActions;
        _temperature = temperature > 0f ? temperature : 1.0f;

        _w1 = new float[inputDim * hiddenDim];
        _b1 = new float[hiddenDim];
        _w2 = new float[hiddenDim * numActions];
        _b2 = new float[numActions];

        InitializeHeWeights();
    }

    private void InitializeHeWeights()
    {
        // He (Kaiming) normal initialization
        var rand = new Random(42);
        float std1 = MathF.Sqrt(2.0f / _inputDim);
        for (int i = 0; i < _w1.Length; i++)
        {
            _w1[i] = (float)(rand.NextDouble() * 2.0 - 1.0) * std1;
        }

        float std2 = MathF.Sqrt(2.0f / _hiddenDim);
        for (int i = 0; i < _w2.Length; i++)
        {
            _w2[i] = (float)(rand.NextDouble() * 2.0 - 1.0) * std2;
        }
    }

    /// <summary>
    /// Evaluates the decision head forward pass with zero heap allocations into caller-provided spans.
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

        // 2. Dense Layer 2: logits = (W2^T * hidden + b2) / Temperature
        float invT = 1.0f / _temperature;
        float maxLogit = float.NegativeInfinity;
        for (int a = 0; a < actions; a++)
        {
            float acc = _b2[a];
            int wOffset = a * _hiddenDim;
            for (int h = 0; h < _hiddenDim; h++)
            {
                acc += hiddenBuffer[h] * _w2[wOffset + h];
            }
            float scaled = acc * invT;
            logitsBuffer[a] = scaled;
            if (scaled > maxLogit)
            {
                maxLogit = scaled;
            }
        }

        // 3. Fused Softmax & In-place Argmax reduction
        float sumExp = 0f;
        for (int a = 0; a < actions; a++)
        {
            float expVal = MathF.Exp(logitsBuffer[a] - maxLogit);
            logitsBuffer[a] = expVal;
            sumExp += expVal;
        }

        float invSum = sumExp > 0f ? 1.0f / sumExp : 1.0f;
        uint bestAction = 0;
        float bestProb = 0f;

        for (int a = 0; a < actions; a++)
        {
            float prob = logitsBuffer[a] * invSum;
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

    public void Dispose() { }
}
