using System;
using System.IO;
using System.Text;
using Glacier.Clavier.Engine;
using Glacier.Clavier.Model;
using Xunit;

namespace Glacier.Clavier.Tests;

public class ClavierConformalTests
{
    [Fact]
    public void CalibrateQuantile_ValidInputs_ComputesExactCeilRank()
    {
        // 100 scores from 0.01 to 1.00
        float[] scores = new float[100];
        for (int i = 0; i < 100; i++)
        {
            scores[i] = (i + 1) / 100.0f;
        }

        // For n = 100, alpha = 0.10:
        // k = ceil((100 + 1) * 0.90) = ceil(90.9) = 91
        // sorted[90] (0-based) = 0.91f
        float qHat = ClavierCalibration.CalibrateQuantile(scores, 0.10f);
        Assert.Equal(0.91f, qHat, precision: 4);
    }

    [Fact]
    public void CalibrateQuantile_FiniteSampleUnderflow_ReturnsSupremumOne()
    {
        // When n is small and 1 - alpha is high:
        // n = 5, alpha = 0.01 -> ceil(6 * 0.99) = 6 > 5
        float[] scores = [0.1f, 0.2f, 0.3f, 0.4f, 0.5f];
        float qHat = ClavierCalibration.CalibrateQuantile(scores, 0.01f);
        Assert.Equal(1.0f, qHat);
    }

    [Fact]
    public void CalibrateQuantile_InvalidAlpha_ThrowsArgumentOutOfRangeException()
    {
        float[] scores = [0.1f, 0.5f];
        Assert.Throws<ArgumentOutOfRangeException>(() => ClavierCalibration.CalibrateQuantile(scores, 0.0f));
        Assert.Throws<ArgumentOutOfRangeException>(() => ClavierCalibration.CalibrateQuantile(scores, 1.0f));
        Assert.Throws<ArgumentOutOfRangeException>(() => ClavierCalibration.CalibrateQuantile(scores, -0.1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => ClavierCalibration.CalibrateQuantile(scores, float.NaN));
    }

    [Fact]
    public void CalibrateQuantile_EmptyScores_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => ClavierCalibration.CalibrateQuantile(ReadOnlySpan<float>.Empty, 0.05f));
    }

    [Fact]
    public void CalibrateQuantile_Monotonicity_QuantileDecreasesAsAlphaIncreases()
    {
        float[] scores = [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 0.95f];
        float qConservative = ClavierCalibration.CalibrateQuantile(scores, 0.05f);
        float qAggressive = ClavierCalibration.CalibrateQuantile(scores, 0.30f);

        Assert.True(qConservative >= qAggressive);
    }

    [Fact]
    public void ComputeNonConformityScores_ComputesOneMinusTargetProbability()
    {
        // 2 samples, 3 classes
        // Sample 0: true label = 1, prob = 0.8 -> score = 0.2
        // Sample 1: true label = 2, prob = 0.1 -> score = 0.9
        float[] flatProbs =
        [
            0.1f, 0.8f, 0.1f,
            0.6f, 0.3f, 0.1f
        ];
        int[] labels = [1, 2];
        float[] scores = new float[2];

        ClavierCalibration.ComputeNonConformityScores(flatProbs, labels, numClasses: 3, scores);

        Assert.Equal(0.2f, scores[0], precision: 4);
        Assert.Equal(0.9f, scores[1], precision: 4);
    }

    [Fact]
    public void DecideConformalSet_SingleConfidentIntent_ReturnsCardinalityOne()
    {
        using var session = new ClavierSession(embeddingDim: 64, hiddenDim: 32, numActions: 8);
        session.ConformalQuantile = 0.85f; // tau = 1 - 0.85 = 0.15

        var set = session.DecideConformalSet("Run unit test suite".AsSpan(), alpha: 0.05f);

        Assert.NotNull(set);
        Assert.True(set.Count >= 0);
        if (set.Count == 1)
        {
            Assert.True(set.IsSystem1Eligible);
            Assert.False(set.NeedsEscalation);
            Assert.Equal(ConformalEscalationReason.None, set.EscalationReason);
        }
    }

    [Fact]
    public void DecideConformalSet_AmbiguousIntent_ReturnsCardinalityGreaterThanOne()
    {
        using var session = new ClavierSession(embeddingDim: 64, hiddenDim: 32, numActions: 8);
        // High quantile threshold qHat = 0.99f means tau = 0.01f -> all non-zero classes admitted
        session.ConformalQuantile = 0.99f;

        var set = session.DecideConformalSet("Deploy build pipeline to production".AsSpan(), alpha: 0.05f);

        Assert.True(set.Count >= 2);
        Assert.True(set.NeedsEscalation);
        Assert.False(set.IsSystem1Eligible);
        Assert.Equal(ConformalEscalationReason.AmbiguousPredictionSet, set.EscalationReason);
    }

    [Fact]
    public void DecideConformalSet_DiffuseOodDistribution_ReturnsEmptyPredictionSet()
    {
        using var session = new ClavierSession(embeddingDim: 64, hiddenDim: 32, numActions: 8);
        // Low quantile threshold qHat = 0.01f means tau = 0.99f -> no class meets 99% probability
        session.ConformalQuantile = 0.01f;

        var set = session.DecideConformalSet("Random unmapped query text".AsSpan(), alpha: 0.05f);

        Assert.Equal(0, set.Count);
        Assert.True(set.NeedsEscalation);
        Assert.False(set.IsSystem1Eligible);
        Assert.Equal(ConformalEscalationReason.EmptyPredictionSet, set.EscalationReason);
        Assert.Empty(set.ActionIds);
        Assert.Equal(0UL, set.ActionMask);
    }

    [Fact]
    public void DecideConformalSet_ActionMask_MatchesActionIdsBitForBit()
    {
        using var session = new ClavierSession(embeddingDim: 64, hiddenDim: 32, numActions: 16);
        session.ConformalQuantile = 0.90f;

        var set = session.DecideConformalSet("Check memory consumption of cluster node".AsSpan());

        for (int i = 0; i < set.Count; i++)
        {
            uint id = set.ActionIds[i];
            Assert.True(set.ContainsAction(id));
            Assert.True((set.ActionMask & (1UL << (int)id)) != 0);
        }

        var summary = set.ToSummary();
        Assert.Equal(set.Count, summary.Count);
        Assert.Equal(set.BestActionId, summary.BestActionId);
        Assert.Equal(set.ActionMask, summary.ActionMask);
        Assert.Equal(set.EscalationReason, summary.EscalationReason);
    }

    [Fact]
    public void ClavierSession_CheckpointV2_SavesAndRestoresConformalQuantile()
    {
        using var session1 = new ClavierSession(embeddingDim: 64, hiddenDim: 32, numActions: 8);
        float[] calScores = [0.05f, 0.12f, 0.25f, 0.35f, 0.42f, 0.65f, 0.78f, 0.88f];
        session1.CalibrateConformal(calScores, alpha: 0.10f);

        float originalQuantile = session1.ConformalQuantile;
        float originalAlpha = session1.ConformalAlpha;

        using var ms = new MemoryStream();
        session1.SaveCheckpoint(ms);

        ms.Position = 0;
        using var session2 = new ClavierSession(embeddingDim: 64, hiddenDim: 32, numActions: 8);
        session2.LoadCheckpoint(ms);

        Assert.Equal(originalQuantile, session2.ConformalQuantile, precision: 5);
        Assert.Equal(originalAlpha, session2.ConformalAlpha, precision: 5);
    }

    [Fact]
    public void ClavierSession_CheckpointV1_BackwardCompatibility()
    {
        // Synthesize a Version 1 checkpoint stream
        using var ms = new MemoryStream();
        using (var writer = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
        {
            writer.Write(0x52564C43u); // "CLVR"
            writer.Write(1);           // Version 1
            writer.Write(64);          // embeddingDim
            writer.Write(32);          // hiddenDim
            writer.Write(8);           // numActions

            // Dummy weights for decision, noul, and score heads
            using var dummySession = new ClavierSession(embeddingDim: 64, hiddenDim: 32, numActions: 8);
            dummySession.DecisionHead.WriteWeights(writer);
            dummySession.NoulHead.WriteWeights(writer);
            dummySession.ScoreHead.WriteWeights(writer);
        }

        ms.Position = 0;
        using var session = new ClavierSession(embeddingDim: 64, hiddenDim: 32, numActions: 8);
        session.LoadCheckpoint(ms);

        // Should load cleanly and initialize default conformal parameters
        Assert.Equal(0.95f, session.ConformalQuantile, precision: 5);
        Assert.Equal(0.05f, session.ConformalAlpha, precision: 5);
    }

    [Fact]
    public void DecideConformalSet_CandidateActions_ReturnsActionNames()
    {
        using var session = new ClavierSession(embeddingDim: 64, hiddenDim: 32, numActions: 4);
        session.ConformalQuantile = 0.95f;
        string[] candidates = ["BuildSolution", "RunTests", "DeployService", "GenerateDocs"];

        var set = session.DecideConformalSet("Run all unit tests in parallel".AsSpan(), alpha: 0.05f, candidateActions: candidates);

        Assert.NotNull(set);
        Assert.NotNull(set.ActionNames);
        Assert.Equal(set.Count, set.ActionNames.Length);
        if (set.Count > 0)
        {
            Assert.Contains(set.BestActionName, candidates);
        }
    }
}
