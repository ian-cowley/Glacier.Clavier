namespace Glacier.Clavier.Tests;

using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Glacier.Clavier.Engine;
using Glacier.Clavier.Model;
using Xunit;

/// <summary>
/// Tier-5 White-Box Adversarial Coverage Hardening for Glacier.Clavier.
/// Stress-tests split conformal prediction calibration quantile bounds,
/// System-1 to System-2 escalation gating sets, and high action-count boundaries (>64 actions).
/// </summary>
public class Tier5ClavierAdversarialCoverageTests
{
    [Theory]
    [InlineData(1, 0.05f, 1.0f)]   // k = ceil(2 * 0.95) = 2 > 1 => returns 1.0f
    [InlineData(1, 0.50f, 0.5f)]   // k = ceil(2 * 0.50) = 1 <= 1 => returns score[0] = 0.5f
    [InlineData(5, 0.01f, 1.0f)]   // k = ceil(6 * 0.99) = 6 > 5 => returns 1.0f
    [InlineData(19, 0.05f, 0.95f)] // k = ceil(20 * 0.95) = 19 <= 19 => returns max score
    [InlineData(20, 0.05f, 0.95f)] // k = ceil(21 * 0.95) = 20 <= 20 => returns score at index 19
    [InlineData(100, 0.01f, 0.99f)]// k = ceil(101 * 0.99) = 100 <= 100 => returns score at index 99
    public void CalibrateQuantile_ExactFiniteSampleBoundaries_MatchesTheoreticalFormula(int n, float alpha, float expectedMinScore)
    {
        float[] scores = new float[n];
        for (int i = 0; i < n; i++)
        {
            scores[i] = (i + 1) / (float)(n + 1);
        }

        float qHat = ClavierCalibration.CalibrateQuantile(scores, alpha);

        int k = (int)MathF.Ceiling((n + 1) * (1.0f - alpha));
        if (k > n)
        {
            Assert.Equal(1.0f, qHat);
        }
        else
        {
            int expectedIndex = k - 1;
            float expectedScore = scores[expectedIndex];
            Assert.Equal(expectedScore, qHat, precision: 5);
        }
        Assert.True(qHat >= expectedMinScore - 1e-4f);
    }

    [Fact]
    public void CalibrateQuantile_MonotonicityAcrossAlphaSpectrum()
    {
        const int n = 200;
        var rng = new Random(42);
        float[] scores = new float[n];
        for (int i = 0; i < n; i++) scores[i] = rng.NextSingle();

        float prevQ = 1.0f;
        // As alpha increases from 0.01 to 0.99, coverage decreases so qHat must be non-increasing
        for (float alpha = 0.01f; alpha <= 0.99f; alpha += 0.02f)
        {
            float q = ClavierCalibration.CalibrateQuantile(scores, alpha);
            Assert.True(q <= prevQ + 1e-6f, $"Monotonicity violation: q({alpha}) = {q} > prevQ = {prevQ}");
            prevQ = q;
        }
    }

    [Fact]
    public void CalibrateQuantile_EmpiricalCoverageGuarantee_UnderExchangeability()
    {
        // Conformal Prediction finite-sample coverage guarantee: P(Y in C(X)) >= 1 - alpha
        const int nCal = 1000;
        const int nTest = 10000;
        const float alpha = 0.05f; // Target: 95% coverage

        var rng = new Random(1337);
        // Generate synthetic non-conformity scores in [0, 1]
        float[] calScores = new float[nCal];
        for (int i = 0; i < nCal; i++) calScores[i] = MathF.Pow(rng.NextSingle(), 2.0f);

        float qHat = ClavierCalibration.CalibrateQuantile(calScores, alpha);

        // Test against independent exchangeable test distribution
        int coveredCount = 0;
        for (int i = 0; i < nTest; i++)
        {
            float testScore = MathF.Pow(rng.NextSingle(), 2.0f);
            if (testScore <= qHat)
            {
                coveredCount++;
            }
        }

        float empiricalCoverage = coveredCount / (float)nTest;
        // Standard error of Bernoulli(0.95) with N=10000 is sqrt(0.95*0.05/10000) ~ 0.0022
        // Lower 3-sigma tolerance is 0.95 - 0.01 = 0.94
        Assert.True(empiricalCoverage >= 0.94f,
            $"Conformal coverage guarantee failed! Target >= 0.95, actual empirical coverage: {empiricalCoverage:P2}");
    }

    [Theory]
    [InlineData(0.0f)]
    [InlineData(1.0f)]
    [InlineData(-0.1f)]
    [InlineData(1.5f)]
    [InlineData(float.NaN)]
    public void CalibrateQuantile_InvalidAlpha_ThrowsArgumentOutOfRangeException(float invalidAlpha)
    {
        float[] scores = [0.1f, 0.2f, 0.3f];
        Assert.Throws<ArgumentOutOfRangeException>(() => ClavierCalibration.CalibrateQuantile(scores, invalidAlpha));
    }

    [Fact]
    public void CalibrateQuantile_EmptyScores_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => ClavierCalibration.CalibrateQuantile(ReadOnlySpan<float>.Empty, 0.05f));
    }

    [Fact]
    public void System1ToSystem2_EscalationGating_StateTransitions_Verified()
    {
        using var session = new ClavierSession(embeddingDim: 64, hiddenDim: 32, numActions: 8);

        // 1. OOD / Diffuse distribution => EmptyPredictionSet => Escalates to System-2
        session.ConformalQuantile = 0.0001f; // tau = 0.9999f
        var emptySet = session.DecideConformalSet("Unseen OOD noise prompt".AsSpan());
        Assert.Equal(0, emptySet.Count);
        Assert.Equal(ConformalEscalationReason.EmptyPredictionSet, emptySet.EscalationReason);
        Assert.True(emptySet.NeedsEscalation);
        Assert.False(emptySet.IsSystem1Eligible);
        Assert.Equal(0UL, emptySet.ActionMask);

        // 2. High uncertainty / Ambiguous => AmbiguousPredictionSet => Escalates to System-2
        session.ConformalQuantile = 1.0f; // tau = 0.0f => all actions included
        var ambigSet = session.DecideConformalSet("Confusing command with multiple valid intents".AsSpan());
        Assert.Equal(8, ambigSet.Count);
        Assert.Equal(ConformalEscalationReason.AmbiguousPredictionSet, ambigSet.EscalationReason);
        Assert.True(ambigSet.NeedsEscalation);
        Assert.False(ambigSet.IsSystem1Eligible);
        Assert.Equal((1UL << 8) - 1UL, ambigSet.ActionMask);

        // 3. Blittable Summary Conversion & Size
        var summary = ambigSet.ToSummary();
        Assert.Equal(8, summary.Count);
        Assert.Equal(ambigSet.BestActionId, summary.BestActionId);
        Assert.Equal(ambigSet.ActionMask, summary.ActionMask);
        Assert.Equal(ConformalEscalationReason.AmbiguousPredictionSet, summary.EscalationReason);
        Assert.True(summary.NeedsEscalation);
        Assert.False(summary.IsSystem1Eligible);
    }

    [Fact]
    public void HighActionCount_BoundaryBeyond64Actions_ActionIdsAndContainsActionCorrect()
    {
        const int numActions = 128;
        string[] candidates = new string[numActions];
        for (int i = 0; i < numActions; i++)
        {
            candidates[i] = $"CandidateAction_{i}";
        }

        using var session = new ClavierSession(embeddingDim: 64, hiddenDim: 32, numActions: 16);
        session.ConformalQuantile = 1.0f; // Admit all actions

        var set = session.DecideConformalSet("Execute cluster deployment".AsSpan(), candidateActions: candidates);

        Assert.Equal(numActions, set.Count);
        Assert.Equal(numActions, set.ActionIds.Length);
        Assert.NotNull(set.ActionNames);
        Assert.Equal(numActions, set.ActionNames.Length);

        // Bitmask is 64-bit, so it only covers actions 0..63
        Assert.Equal(ulong.MaxValue, set.ActionMask);

        // Verify actions at the exact 64 boundary:
        Assert.True(set.ContainsAction(63u));
        Assert.True(set.ContainsAction(64u));
        Assert.True(set.ContainsAction(65u));
        Assert.True(set.ContainsAction(127u));
        Assert.False(set.ContainsAction(128u));
        Assert.False(set.ContainsAction(999u));
    }

    [Fact]
    public void HighActionCount_LargeActionSet_ArrayPoolSharedPath_ConcurrentFuzzing()
    {
        // 320 candidate actions forces the ArrayPool<uint>.Shared rented path (> 256)
        const int numActions = 320;
        string[] candidates = new string[numActions];
        for (int i = 0; i < numActions; i++)
        {
            candidates[i] = $"OpCode_{i}";
        }

        using var session = new ClavierSession(embeddingDim: 64, hiddenDim: 32, numActions: 16);
        session.ConformalQuantile = 1.0f;

        // Run 50 concurrent requests to stress-test ArrayPool rent/return safety
        Parallel.For(0, 50, iter =>
        {
            var set = session.DecideConformalSet("Parallel high-count action decision".AsSpan(), candidateActions: candidates);
            Assert.Equal(numActions, set.Count);
            Assert.Equal(numActions, set.ActionIds.Length);
            Assert.True(set.ContainsAction(0u));
            Assert.True(set.ContainsAction(255u));
            Assert.True(set.ContainsAction(256u));
            Assert.True(set.ContainsAction(319u));
            Assert.False(set.ContainsAction(320u));
        });
    }

    [Fact]
    public void ClavierSession_CheckpointRoundtrip_PreservesConformalQuantileAndScores()
    {
        using var session = new ClavierSession(embeddingDim: 64, hiddenDim: 32, numActions: 16);
        float[] holdoutScores = [0.12f, 0.45f, 0.67f, 0.89f, 0.94f];
        session.CalibrateConformal(holdoutScores, alpha: 0.10f);

        float originalQuantile = session.ConformalQuantile;
        float originalAlpha = session.ConformalAlpha;

        using var ms = new MemoryStream();
        session.SaveCheckpoint(ms);

        ms.Position = 0;
        using var loadedSession = new ClavierSession(embeddingDim: 64, hiddenDim: 32, numActions: 16);
        loadedSession.LoadCheckpoint(ms);

        Assert.Equal(originalQuantile, loadedSession.ConformalQuantile, precision: 5);
        Assert.Equal(originalAlpha, loadedSession.ConformalAlpha, precision: 5);
    }
}
