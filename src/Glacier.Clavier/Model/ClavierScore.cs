namespace Glacier.Clavier.Model;

using System;
using System.Runtime.InteropServices;

/// <summary>
/// Blittable 8-byte result for the Jev 'Score' continuous scalar evaluation primitive.
/// Zero-copy transfer across PCIe / Unified Memory.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1, Size = 8)]
public readonly struct ClavierScore
{
    private readonly float _value;
    private readonly float _confidence;

    public float Value => _value;
    public float Confidence => _confidence;

    public ClavierScore(float value, float confidence)
    {
        _value = value;
        _confidence = confidence;
    }

    public override string ToString() =>
        $"Score(Value: {Value:F2}, Conf: {Confidence * 100f:F1}%)";
}

/// <summary>
/// High-level evaluation result for the 'Score' primitive with scale bounds and latency.
/// </summary>
public sealed record ClavierScoreResult
{
    public required ClavierScore Score { get; init; }
    public required float Value { get; init; }
    public required float Confidence { get; init; }
    public required float Min { get; init; }
    public required float Max { get; init; }
    public required double LatencyMs { get; init; }
}
