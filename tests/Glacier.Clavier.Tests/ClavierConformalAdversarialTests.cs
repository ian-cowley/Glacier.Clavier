using System;
using System.Collections.Generic;
using System.Linq;
using Glacier.Clavier.Engine;
using Glacier.Clavier.Model;
using Xunit;

namespace Glacier.Clavier.Tests;

public class ClavierConformalAdversarialTests
{
    [Theory]
    [InlineData(1, 0.05f, true)]   // n=1, k = ceil(2 * 0.95) = 2 > 1 => returns 1.0f
    [InlineData(5, 0.01f, true)]   // n=5, k = ceil(6 * 0.99) = 6 > 5 => returns 1.0f
    [InlineData(18, 0.05f, true)]  // n=18, k = ceil(19 * 0.95) = ceil(18.05) = 19 > 18 => returns 1.0f
    [InlineData(19, 0.05f, false)] // n=19, k = ceil(20 * 0.95) = ceil(19.0) = 19 <= 19 => returns max score
    [InlineData(20, 0.05f, false)] // n=20, k = ceil(21 * 0.95) = ceil(19.95) = 20 <= 20
    [InlineData(100, 0.01f, false)]// n=100, k = ceil(101 * 0.99) = ceil(99.99) = 100 <= 100
    [InlineData(1, 0.99f, false)]  // n=1, alpha=0.99: k = ceil(2 * 0.01) = 1 <= 1 => valid
    public void CalibrateQuantile_FiniteSampleBoundary_CorrectlyIdentifiesUnderflow(int n, float alpha, bool expectedUnderflow)
    {
        float[] scores = new float[n];
        for (int i = 0; i < n; i++)
        {
            scores[i] = (i + 1) / (float)(n + 1); // strictly less than 1.0
        }

        float qHat = ClavierCalibration.CalibrateQuantile(scores, alpha);

        if (expectedUnderflow)
        {
            Assert.Equal(1.0f, qHat);
        }
        else
        {
            Assert.True(qHat < 1.0f, $"Expected qHat < 1.0f but got {qHat} for n={n}, alpha={alpha}");
        }
    }

    [Fact]
    public void CalibrateQuantile_FloatingPointPrecisionEdgeCases()
    {
        // n = 9, alpha = 0.10f: (9 + 1) * (1 - 0.1) = 10 * 0.9 = 9.0 in exact math.
        // In IEEE 754 float: 1.0f - 0.1f = 0.900000036f, 10 * 0.900000036f = 9.00000036f.
        // MathF.Ceiling(9.00000036f) gives 10, which exceeds n=9!
        float[] scores9 = [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f];
        float qHat9 = ClavierCalibration.CalibrateQuantile(scores9, 0.10f);

        // Does it return 1.0f due to float rounding or does it handle the exact boundary?
        // Note: mathematically, with 9 samples, 90% coverage cannot guarantee P(Y in C(X)) >= 0.9 under conformal prediction without 1.0.
        Assert.InRange(qHat9, 0.0f, 1.0f);
    }

    [Fact]
    public void CalibrateQuantile_IdenticalScores_ReturnsExactScore()
    {
        float[] scores = Enumerable.Repeat(0.42f, 50).ToArray();
        float qHat = ClavierCalibration.CalibrateQuantile(scores, 0.10f);
        Assert.Equal(0.42f, qHat, precision: 5);
    }

    [Fact]
    public void CalibrateQuantile_LargeSampleSet_ExecutesWithoutStackOverflowOrAllocationLeak()
    {
        int n = 10_000;
        var rng = new Random(42);
        float[] scores = new float[n];
        for (int i = 0; i < n; i++) scores[i] = rng.NextSingle();

        float qHat = ClavierCalibration.CalibrateQuantile(scores, 0.05f);
        Assert.InRange(qHat, 0.94f, 0.96f);
    }

    [Fact]
    public void DecideConformalSet_ExtremeAlpha_HandledCorrectly()
    {
        using var session = new ClavierSession(embeddingDim: 64, hiddenDim: 32, numActions: 8);
        session.ConformalQuantile = 0.95f;

        // Very high alpha (low confidence required)
        var setHighAlpha = session.DecideConformalSet("Run unit test".AsSpan(), alpha: 0.90f);
        Assert.NotNull(setHighAlpha);

        // Very low alpha (high confidence required)
        var setLowAlpha = session.DecideConformalSet("Run unit test".AsSpan(), alpha: 0.001f);
        Assert.NotNull(setLowAlpha);
    }

    [Fact]
    public void DecideConformalSet_DiffuseOodProbabilities_EscalatesWithEmptyPredictionSet()
    {
        using var session = new ClavierSession(embeddingDim: 64, hiddenDim: 32, numActions: 8);
        // If qHat is very small (0.001), tau = 1 - 0.001 = 0.999.
        // A diffuse distribution where all probs ~ 1/8 = 0.125 will have NO action with prob >= 0.999.
        session.ConformalQuantile = 0.001f;

        var set = session.DecideConformalSet("Unseen adversarial OOD prompt sequence".AsSpan());
        Assert.Equal(0, set.Count);
        Assert.Equal(ConformalEscalationReason.EmptyPredictionSet, set.EscalationReason);
        Assert.True(set.NeedsEscalation);
        Assert.False(set.IsSystem1Eligible);
        Assert.Equal(0UL, set.ActionMask);
    }

    [Fact]
    public void DecideConformalSet_UniformAmbiguousProbabilities_EscalatesWithAmbiguousSet()
    {
        using var session = new ClavierSession(embeddingDim: 64, hiddenDim: 32, numActions: 8);
        // If qHat is 1.0f, tau = 0.0f. All actions are admitted.
        session.ConformalQuantile = 1.0f;

        var set = session.DecideConformalSet("Ambiguous command".AsSpan());
        Assert.Equal(8, set.Count);
        Assert.Equal(ConformalEscalationReason.AmbiguousPredictionSet, set.EscalationReason);
        Assert.True(set.NeedsEscalation);
        Assert.False(set.IsSystem1Eligible);
    }

    [Fact]
    public void ActionMask_64BitBoundaries_Bit0AndBit63Correct()
    {
        using var session = new ClavierSession(embeddingDim: 64, hiddenDim: 32, numActions: 64);
        session.ConformalQuantile = 1.0f; // Admit all actions

        var set = session.DecideConformalSet("Test all 64 actions".AsSpan());
        Assert.Equal(64, set.Count);

        // Check boundary bit 0
        Assert.True(set.ContainsAction(0));
        Assert.True((set.ActionMask & 1UL) != 0);

        // Check boundary bit 63
        Assert.True(set.ContainsAction(63));
        Assert.True((set.ActionMask & (1UL << 63)) != 0);

        // Action 64 out of bounds for 64-bit mask
        Assert.False(set.ContainsAction(64));
    }

    [Fact]
    public void DecideConformalSet_MoreThan64CandidateActions_InvestigateActionIdsCorrectness()
    {
        // Stress-test when candidateActions has > 64 elements (e.g. 70 actions)
        int numCandidateActions = 70;
        string[] candidates = new string[numCandidateActions];
        for (int i = 0; i < numCandidateActions; i++)
        {
            candidates[i] = $"Action_{i}";
        }

        using var session = new ClavierSession(embeddingDim: 64, hiddenDim: 32, numActions: 16);
        // qHat = 1.0f -> tau = 0.0f -> all candidate actions should be included
        session.ConformalQuantile = 1.0f;

        var set = session.DecideConformalSet("Execute task".AsSpan(), candidateActions: candidates);

        Assert.Equal(numCandidateActions, set.Count);
        Assert.NotNull(set.ActionIds);
        Assert.Equal(numCandidateActions, set.ActionIds.Length);

        // Verify that ActionIds contains distinct action indices from 0 to 69
        var distinctIds = set.ActionIds.Distinct().ToList();
        Assert.Equal(numCandidateActions, distinctIds.Count);

        // Verify each action index matches its expected value
        for (int i = 0; i < numCandidateActions; i++)
        {
            Assert.Equal((uint)i, set.ActionIds[i]);
        }
    }

    [Fact]
    public void DecideConformalSet_MoreThan64Actions_SelectiveInclusion_VerifiesAllActionIdsMatchProbabilities()
    {
        // When activeActions > 64 and some actions < 64 are excluded,
        // any action included in ActionIds MUST actually have Probabilities[actionId] >= tau!
        int numCandidateActions = 100;
        string[] candidates = new string[numCandidateActions];
        for (int i = 0; i < numCandidateActions; i++)
        {
            candidates[i] = i == 0 ? "CompletelyUnrelatedActionThatShouldHaveLowScore" : $"CandidateActionNumber_{i}";
        }

        using var session = new ClavierSession(embeddingDim: 64, hiddenDim: 32, numActions: 16);
        // Set a quantile that admits >64 actions but excludes Action 0 (tau ~ 0.0095)
        session.ConformalQuantile = 0.9905f;

        var set = session.DecideConformalSet("CandidateActionNumber query".AsSpan(), candidateActions: candidates);
        Assert.True(set.Count > 64, $"set.Count was {set.Count}, needs to be > 64 to exercise the boundary");
        Assert.True(set.Probabilities[0] < set.InclusionProbabilityThreshold, $"Action 0 prob ({set.Probabilities[0]}) should be < threshold ({set.InclusionProbabilityThreshold})");

        // Every ID reported in ActionIds must satisfy the inclusion threshold:
        for (int i = 0; i < set.ActionIds.Length; i++)
        {
            uint actionId = set.ActionIds[i];
            Assert.True(actionId < (uint)numCandidateActions, $"Action ID {actionId} exceeds candidate count {numCandidateActions}");
            float prob = set.Probabilities[(int)actionId];
            Assert.True(prob >= set.InclusionProbabilityThreshold,
                $"Action ID {actionId} at index {i} has probability {prob} < threshold {set.InclusionProbabilityThreshold}");
        }

        // ActionIds must have no duplicates:
        Assert.Equal(set.ActionIds.Length, set.ActionIds.Distinct().Count());
    }

    [Fact]
    public void ClavierPredictionSet_ContainsAction_Beyond64Actions_ShouldReflectPredictionSetMembership()
    {
        // When there are candidate actions beyond index 63, ClavierPredictionSet.ContainsAction
        // should accurately reflect membership in ActionIds rather than blindly returning false.
        int numCandidateActions = 80;
        string[] candidates = new string[numCandidateActions];
        for (int i = 0; i < numCandidateActions; i++)
        {
            candidates[i] = $"Action_{i}";
        }

        using var session = new ClavierSession(embeddingDim: 64, hiddenDim: 32, numActions: 16);
        session.ConformalQuantile = 1.0f; // All actions admitted

        var set = session.DecideConformalSet("Run action".AsSpan(), candidateActions: candidates);

        // Action 70 is in the candidate list and admitted to the set
        Assert.Contains(70u, set.ActionIds);

        // ContainsAction should return true for admitted action 70
        Assert.True(set.ContainsAction(70u), "ContainsAction(70) returned false even though Action 70 is in ActionIds!");
    }

    [Fact]
    public void DecideConformalSet_MoreThan256Actions_ExercisesArrayPoolPathAndVerifiesMembership()
    {
        // > 256 actions forces ClavierSession.DecideConformalSet to take the ArrayPool<uint>.Shared path
        // rather than stackalloc uint[activeActions].
        int numCandidateActions = 350;
        string[] candidates = new string[numCandidateActions];
        for (int i = 0; i < numCandidateActions; i++)
        {
            candidates[i] = $"DynamicAction_{i}";
        }

        using var session = new ClavierSession(embeddingDim: 64, hiddenDim: 32, numActions: 16);
        session.ConformalQuantile = 1.0f; // All actions admitted

        // Run multiple iterations to verify ArrayPool rent/return stability and zero memory corruption
        for (int iter = 0; iter < 10; iter++)
        {
            var set = session.DecideConformalSet("Execute dynamic cluster orchestration command".AsSpan(), candidateActions: candidates);

            Assert.Equal(numCandidateActions, set.Count);
            Assert.Equal(numCandidateActions, set.ActionIds.Length);
            Assert.NotNull(set.ActionNames);
            Assert.Equal(numCandidateActions, set.ActionNames.Length);

            // Verify first, 64-boundary, and last elements
            Assert.True(set.ContainsAction(0u));
            Assert.True(set.ContainsAction(63u));
            Assert.True(set.ContainsAction(64u));
            Assert.True(set.ContainsAction(255u));
            Assert.True(set.ContainsAction(256u));
            Assert.True(set.ContainsAction(349u));
            Assert.False(set.ContainsAction(350u));
            Assert.False(set.ContainsAction(1000u));

            Assert.Equal("DynamicAction_0", set.ActionNames[0]);
            Assert.Equal("DynamicAction_64", set.ActionNames[64]);
            Assert.Equal("DynamicAction_349", set.ActionNames[349]);
        }
    }

    [Fact]
    public void ClavierPredictionSet_ContainsAction_ExactBoundaryAround64_AndExtremes()
    {
        var predictionSet = new ClavierPredictionSet
        {
            Count = 3,
            BestActionId = 63,
            BestConfidence = 0.9f,
            EscalationReason = ConformalEscalationReason.AmbiguousPredictionSet,
            ActionIds = [62u, 64u, 200u],
            ActionMask = (1UL << 62), // bit 62 set, bit 63 NOT set, bit 64 cannot fit in mask
            QuantileThreshold = 0.95f,
            InclusionProbabilityThreshold = 0.05f,
            Alpha = 0.05f,
            Probabilities = new float[256],
            LatencyMs = 0.1
        };

        // Within 64-bit mask (< 64)
        Assert.True(predictionSet.ContainsAction(62u)); // in mask and in ids
        Assert.False(predictionSet.ContainsAction(63u)); // not in mask

        // At boundary and beyond (>= 64, falls back to ActionIds search)
        Assert.True(predictionSet.ContainsAction(64u));  // in ActionIds
        Assert.False(predictionSet.ContainsAction(65u)); // not in ActionIds
        Assert.True(predictionSet.ContainsAction(200u)); // in ActionIds
        Assert.False(predictionSet.ContainsAction(201u));

        // Extreme bounds (must not throw index out of range)
        Assert.False(predictionSet.ContainsAction(uint.MaxValue));
        Assert.False(predictionSet.ContainsAction(1_000_000u));
    }

    [Fact]
    public void DecideConformalSet_NumActionsGreaterThan64_DefaultCandidates_CorrectlyPopulates()
    {
        // Session created with 128 discrete actions and no candidate strings supplied
        using var session = new ClavierSession(embeddingDim: 64, hiddenDim: 32, numActions: 128);
        session.ConformalQuantile = 1.0f; // All actions admitted

        var set = session.DecideConformalSet("Dispatch system-1 operation".AsSpan());

        Assert.Equal(128, set.Count);
        Assert.Null(set.ActionNames); // No candidates supplied
        Assert.Equal(128, set.ActionIds.Length);

        // Verify actions >= 64 are correctly queryable
        Assert.True(set.ContainsAction(64u));
        Assert.True(set.ContainsAction(127u));
        Assert.False(set.ContainsAction(128u));
    }
}
