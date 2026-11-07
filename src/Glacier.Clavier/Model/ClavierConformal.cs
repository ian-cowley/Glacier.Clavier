namespace Glacier.Clavier.Model;

using System;
using System.Runtime.InteropServices;

/// <summary>
/// Specifies the formal gating status of a split conformal prediction set.
/// </summary>
public enum ConformalEscalationReason : byte
{
    /// <summary>
    /// |C(x)| == 1: Exactly one action meets threshold; eligible for sub-11 µs System-1 fast dispatch.
    /// </summary>
    None = 0,

    /// <summary>
    /// |C(x)| == 0: Empty prediction set. Softmax distribution is diffuse / out-of-distribution.
    /// Escalates immediately to System-2 LLM.
    /// </summary>
    EmptyPredictionSet = 1,

    /// <summary>
    /// |C(x)| > 1: Ambiguous prediction set containing multiple conflicting candidate actions.
    /// Escalates immediately to System-2 LLM with candidates for disambiguation.
    /// </summary>
    AmbiguousPredictionSet = 2
}

/// <summary>
/// 24-byte unmanaged, blittable summary of conformal prediction set gating.
/// Transferred across call boundaries with zero heap allocation.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 4)]
public readonly struct ClavierPredictionSetSummary : IEquatable<ClavierPredictionSetSummary>
{
    /// <summary>
    /// Cardinality of the prediction set |C(x)|.
    /// </summary>
    public readonly int Count;

    /// <summary>
    /// Argmax discrete action index.
    /// </summary>
    public readonly uint BestActionId;

    /// <summary>
    /// Argmax calibrated softmax confidence probability in [0.0, 1.0].
    /// </summary>
    public readonly float BestConfidence;

    /// <summary>
    /// Bitmask representing membership of actions 0..63 in prediction set C(x).
    /// </summary>
    public readonly ulong ActionMask;

    /// <summary>
    /// Calibrated conformal non-conformity quantile threshold qHat.
    /// </summary>
    public readonly float QuantileThreshold;

    /// <summary>
    /// Gating escalation status.
    /// </summary>
    public readonly ConformalEscalationReason EscalationReason;

    public bool NeedsEscalation => Count != 1;
    public bool IsSystem1Eligible => Count == 1;

    public ClavierPredictionSetSummary(
        int count,
        uint bestActionId,
        float bestConfidence,
        ulong actionMask,
        float quantileThreshold,
        ConformalEscalationReason escalationReason)
    {
        Count = count;
        BestActionId = bestActionId;
        BestConfidence = bestConfidence;
        ActionMask = actionMask;
        QuantileThreshold = quantileThreshold;
        EscalationReason = escalationReason;
    }

    public bool ContainsAction(uint actionId) =>
        actionId < 64 && (ActionMask & (1UL << (int)actionId)) != 0;

    public bool Equals(ClavierPredictionSetSummary other) =>
        Count == other.Count &&
        BestActionId == other.BestActionId &&
        MathF.Abs(BestConfidence - other.BestConfidence) < 1e-6f &&
        ActionMask == other.ActionMask &&
        MathF.Abs(QuantileThreshold - other.QuantileThreshold) < 1e-6f &&
        EscalationReason == other.EscalationReason;

    public override bool Equals(object? obj) => obj is ClavierPredictionSetSummary other && Equals(other);
    public override int GetHashCode() => HashCode.Combine(Count, BestActionId, ActionMask, (int)EscalationReason);
    public static bool operator ==(ClavierPredictionSetSummary left, ClavierPredictionSetSummary right) => left.Equals(right);
    public static bool operator !=(ClavierPredictionSetSummary left, ClavierPredictionSetSummary right) => !left.Equals(right);
}

/// <summary>
/// Detailed evaluation of a split conformal prediction set C(x) including calibrated probabilities,
/// candidate action names, bitmask, and escalation diagnostics.
/// </summary>
public sealed class ClavierPredictionSet
{
    /// <summary>
    /// Cardinality of the prediction set |C(x)|.
    /// </summary>
    public required int Count { get; init; }

    /// <summary>
    /// True when |C(x)| != 1 (either empty or ambiguous), requiring System-2 LLM execution.
    /// </summary>
    public bool NeedsEscalation => Count != 1;

    /// <summary>
    /// True when |C(x)| == 1, authorizing fast sub-11 µs System-1 execution.
    /// </summary>
    public bool IsSystem1Eligible => Count == 1;

    /// <summary>
    /// Formal reason for escalation (None, EmptyPredictionSet, AmbiguousPredictionSet).
    /// </summary>
    public required ConformalEscalationReason EscalationReason { get; init; }

    /// <summary>
    /// Argmax discrete action index.
    /// </summary>
    public required uint BestActionId { get; init; }

    /// <summary>
    /// Name of the argmax action if candidate action names were provided.
    /// </summary>
    public string? BestActionName { get; init; }

    /// <summary>
    /// Calibrated softmax confidence probability of the winning action.
    /// </summary>
    public required float BestConfidence { get; init; }

    /// <summary>
    /// Indices of all actions included in prediction set C(x).
    /// </summary>
    public required uint[] ActionIds { get; init; }

    /// <summary>
    /// Names of actions included in prediction set C(x) (when candidate actions are supplied).
    /// </summary>
    public string[]? ActionNames { get; init; }

    /// <summary>
    /// 64-bit membership bitmask.
    /// </summary>
    public required ulong ActionMask { get; init; }

    /// <summary>
    /// Calibrated non-conformity threshold qHat in [0.0, 1.0].
    /// </summary>
    public required float QuantileThreshold { get; init; }

    /// <summary>
    /// Minimum required probability threshold for set inclusion: tau = 1 - qHat.
    /// </summary>
    public required float InclusionProbabilityThreshold { get; init; }

    /// <summary>
    /// Significance level alpha in (0, 1) used for calibration.
    /// </summary>
    public required float Alpha { get; init; }

    /// <summary>
    /// Full calibrated softmax probability distribution across actions.
    /// </summary>
    public required float[] Probabilities { get; init; }

    /// <summary>
    /// Total forward pass and conformal evaluation latency in milliseconds.
    /// </summary>
    public double LatencyMs { get; init; }

    public bool ContainsAction(uint actionId) =>
        actionId < 64
            ? (ActionMask & (1UL << (int)actionId)) != 0
            : (ActionIds != null && Array.IndexOf(ActionIds, actionId) >= 0);

    public ClavierPredictionSetSummary ToSummary() => new(
        Count,
        BestActionId,
        BestConfidence,
        ActionMask,
        QuantileThreshold,
        EscalationReason);
}
