namespace Glacier.Clavier.Model;

using System;
using System.Runtime.InteropServices;

/// <summary>
/// Blittable 8-byte result for the Jev 'Noul' binary verification primitive.
/// Zero-copy transfer across PCIe / Unified Memory.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1, Size = 8)]
public readonly struct ClavierNoul
{
    private readonly byte _affirmative;
    private readonly byte _reserved1;
    private readonly byte _reserved2;
    private readonly byte _reserved3;
    private readonly float _probability;

    public bool IsAffirmative => _affirmative != 0;
    public float Probability => _probability;
    public float Confidence => MathF.Abs(_probability - 0.5f) * 2.0f; // Certainty scale [0, 1]

    public ClavierNoul(bool isAffirmative, float probability)
    {
        _affirmative = isAffirmative ? (byte)1 : (byte)0;
        _reserved1 = 0;
        _reserved2 = 0;
        _reserved3 = 0;
        _probability = probability;
    }

    public override string ToString() =>
        $"Noul(Affirmative: {IsAffirmative}, Prob: {Probability * 100f:F1}%, Conf: {Confidence * 100f:F1}%)";
}

/// <summary>
/// High-level evaluation result for the 'Noul' primitive with latency metrics.
/// </summary>
public sealed record ClavierNoulResult
{
    public required ClavierNoul Noul { get; init; }
    public required bool IsAffirmative { get; init; }
    public required float Probability { get; init; }
    public required float Confidence { get; init; }
    public required double LatencyMs { get; init; }
}
