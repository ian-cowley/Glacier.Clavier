namespace Glacier.Clavier.Model;

using System;
using System.Runtime.InteropServices;

/// <summary>
/// Blittable 8-byte result for the Jev 'Noul' binary verification primitive.
/// Zero-copy transfer across PCIe / Unified Memory with natural 8-byte alignment.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct ClavierNoul : IEquatable<ClavierNoul>
{
    private readonly float _probability;
    private readonly byte _affirmative;

    public bool IsAffirmative => _affirmative != 0;
    public float Probability => _probability;
    public float Confidence => MathF.Abs(_probability - 0.5f) * 2.0f; // Certainty scale [0, 1]

    public ClavierNoul(bool isAffirmative, float probability)
    {
        _affirmative = isAffirmative ? (byte)1 : (byte)0;
        _probability = probability;
    }

    public bool Equals(ClavierNoul other) =>
        _affirmative == other._affirmative && _probability.Equals(other._probability);

    public override bool Equals(object? obj) => obj is ClavierNoul other && Equals(other);
    public override int GetHashCode() => HashCode.Combine(_affirmative, _probability);
    public static bool operator ==(ClavierNoul left, ClavierNoul right) => left.Equals(right);
    public static bool operator !=(ClavierNoul left, ClavierNoul right) => !left.Equals(right);

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
