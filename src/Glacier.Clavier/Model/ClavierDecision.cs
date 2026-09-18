namespace Glacier.Clavier.Model;

using System;
using System.Runtime.InteropServices;

/// <summary>
/// 8-byte unmanaged, blittable decision result transferred directly across PCIe from GPU device memory.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 4)]
public readonly struct ClavierDecision : IEquatable<ClavierDecision>
{
    /// <summary>
    /// Index of the winning discrete action (0..K-1).
    /// </summary>
    public readonly uint ActionId;

    /// <summary>
    /// Calibrated confidence probability in [0.0, 1.0].
    /// </summary>
    public readonly float Confidence;

    public ClavierDecision(uint actionId, float confidence)
    {
        ActionId = actionId;
        Confidence = confidence;
    }

    public bool Equals(ClavierDecision other) => ActionId == other.ActionId && MathF.Abs(Confidence - other.Confidence) < 1e-6f;
    public override bool Equals(object? obj) => obj is ClavierDecision other && Equals(other);
    public override int GetHashCode() => HashCode.Combine(ActionId, Confidence);
    public static bool operator ==(ClavierDecision left, ClavierDecision right) => left.Equals(right);
    public static bool operator !=(ClavierDecision left, ClavierDecision right) => !left.Equals(right);
    public override string ToString() => $"ActionId: {ActionId}, Confidence: {Confidence * 100f:F1}%";
}

/// <summary>
/// Detailed decision evaluation including the full calibrated categorical probability distribution.
/// </summary>
public sealed class ClavierDecisionResult
{
    public required ClavierDecision Decision { get; init; }
    public string? ActionName { get; init; }
    public required float[] Probabilities { get; init; }
    public double LatencyMs { get; init; }
}
