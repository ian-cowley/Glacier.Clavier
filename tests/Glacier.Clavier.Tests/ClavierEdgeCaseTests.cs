namespace Glacier.Clavier.Tests;

using System;
using System.IO;
using System.Threading.Tasks;
using Glacier.Clavier.Engine;
using Glacier.Clavier.Model;
using Xunit;

public class ClavierEdgeCaseTests
{
    [Fact]
    public void ClavierSession_Constructor_ValidatesArguments()
    {
        Assert.Throws<ArgumentException>(() => new ClavierSession(embeddingDim: 0, hiddenDim: 128, numActions: 4));
        Assert.Throws<ArgumentException>(() => new ClavierSession(embeddingDim: 128, hiddenDim: 0, numActions: 4));
        Assert.Throws<ArgumentException>(() => new ClavierSession(embeddingDim: 128, hiddenDim: 128, numActions: 0));

        // When temperature <= 0, head defaults to 1.0f
        var head = new ClavierDecisionHead(128, 64, 4, temperature: -1.0f);
        Assert.Equal(1.0f, head.Temperature);

        // Property setter throws on non-positive temperature
        Assert.Throws<ArgumentOutOfRangeException>(() => head.Temperature = 0.0f);
    }

    [Fact]
    public void ClavierSession_Decide_EmptyActions_FallsBackToConfiguredNumActions()
    {
        using var session = new ClavierSession(embeddingDim: 128, hiddenDim: 64, numActions: 4);
        string[] emptyActions = [];

        var decision = session.Decide("state".AsSpan(), emptyActions);
        Assert.True(decision.ActionId < 4);
        Assert.InRange(decision.Confidence, 0.0f, 1.0f);
    }

    [Fact]
    public void ClavierSession_Decide_EmptyState_ExecutesSuccessfully()
    {
        using var session = new ClavierSession(embeddingDim: 128, hiddenDim: 64, numActions: 4);
        string[] actions = ["ACT_A", "ACT_B"];

        var decision = session.Decide(ReadOnlySpan<char>.Empty, actions);
        Assert.True(decision.ActionId < (uint)actions.Length);
        Assert.InRange(decision.Confidence, 0.0f, 1.0f);
    }

    [Fact]
    public void ClavierSession_Disposal_IsIdempotent()
    {
        var session = new ClavierSession(embeddingDim: 128, hiddenDim: 64, numActions: 4);
        session.Dispose();
        session.Dispose(); // second call should not throw
    }

    [Fact]
    public void ClavierSession_AfterDispose_ThrowsObjectDisposedException()
    {
        var session = new ClavierSession(embeddingDim: 128, hiddenDim: 64, numActions: 4);
        session.Dispose();

        string[] actions = ["A", "B"];
        Assert.Throws<ObjectDisposedException>(() => session.Decide("state".AsSpan(), actions));
    }

    [Fact]
    public void ClavierSession_ConcurrentDecide_IsThreadSafe()
    {
        using var session = new ClavierSession(embeddingDim: 128, hiddenDim: 64, numActions: 8);
        string[] actions = ["N", "S", "E", "W"];

        Parallel.For(0, 50, i =>
        {
            string state = $"Agent #{i} at coordinate ({i * 2}, {i * 3})";
            var dec = session.Decide(state.AsSpan(), actions);
            Assert.True(dec.ActionId < (uint)actions.Length);
            Assert.InRange(dec.Confidence, 0.0f, 1.0f);
        });
    }

    [Fact]
    public void ClavierSession_CandidateActionPermutation_PreservesSemanticWinner()
    {
        using var session = new ClavierSession(embeddingDim: 128, hiddenDim: 64, numActions: 4);
        string state = "Weapon magazine empty (0/30). Hostile unit advancing.";

        string[] orderA = ["ENGAGE_TARGET", "RELOAD_WEAPON", "HOLD_POSITION"];
        string[] orderB = ["HOLD_POSITION", "ENGAGE_TARGET", "RELOAD_WEAPON"];

        var resA = session.DecideWithDistribution(state.AsSpan(), orderA);
        var resB = session.DecideWithDistribution(state.AsSpan(), orderB);

        Assert.Equal(resA.ActionName, resB.ActionName);
        Assert.Equal(resA.Decision.Confidence, resB.Decision.Confidence, precision: 4);
    }

    [Fact]
    public void ClavierSession_OnlineTrainingAndCheckpointRoundtrip_ConvergesAndPersists()
    {
        using var session = new ClavierSession(embeddingDim: 128, hiddenDim: 64, numActions: 4);
        string state = "Magazine empty, hostile approaching at 10m.";
        string[] actions = ["ENGAGE_TARGET", "RELOAD_WEAPON", "SCAN_PERIMETER"];
        string hyp = "Immediate reload required";
        string crit = "Combat urgency";

        float firstChoiceLoss = session.TrainDecision(state.AsSpan(), targetActionIndex: 1, actions, learningRate: 0.1f);
        float lastChoiceLoss = firstChoiceLoss;
        for (int epoch = 0; epoch < 25; epoch++)
        {
            lastChoiceLoss = session.TrainDecision(state.AsSpan(), targetActionIndex: 1, actions, learningRate: 0.1f);
            session.TrainVerify(state.AsSpan(), hyp.AsSpan(), targetAffirmative: true, learningRate: 0.1f);
            session.TrainScore(state.AsSpan(), crit.AsSpan(), targetValue: 90.0f, min: 0.0f, max: 100.0f, learningRate: 0.15f);
        }

        Assert.True(lastChoiceLoss < firstChoiceLoss);

        var trainedChoice = session.DecideWithDistribution(state.AsSpan(), actions);
        Assert.Equal(1u, trainedChoice.Decision.ActionId);
        Assert.Equal("RELOAD_WEAPON", trainedChoice.ActionName);

        var trainedNoul = session.Verify(state.AsSpan(), hyp.AsSpan(), threshold: 0.5f);
        Assert.True(trainedNoul.IsAffirmative);
        Assert.True(trainedNoul.Probability > 0.7f);

        var trainedScore = session.Score(state.AsSpan(), crit.AsSpan(), min: 0.0f, max: 100.0f);
        Assert.InRange(trainedScore.Value, 75.0f, 100.0f);

        // Save and reload into fresh session
        using var ms = new MemoryStream();
        session.SaveCheckpoint(ms);

        ms.Position = 0;
        using var restored = new ClavierSession(embeddingDim: 128, hiddenDim: 64, numActions: 4);
        restored.LoadCheckpoint(ms);

        var restoredChoice = restored.DecideWithDistribution(state.AsSpan(), actions);
        var restoredNoul = restored.Verify(state.AsSpan(), hyp.AsSpan(), threshold: 0.5f);
        var restoredScore = restored.Score(state.AsSpan(), crit.AsSpan(), min: 0.0f, max: 100.0f);

        Assert.Equal(trainedChoice.Decision.ActionId, restoredChoice.Decision.ActionId);
        Assert.Equal(trainedChoice.Decision.Confidence, restoredChoice.Decision.Confidence, precision: 5);
        Assert.Equal(trainedNoul.Probability, restoredNoul.Probability, precision: 5);
        Assert.Equal(trainedScore.Value, restoredScore.Value, precision: 4);
    }
}
