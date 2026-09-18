namespace Glacier.Clavier.Tests;

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Glacier.Clavier.Engine;
using Glacier.Clavier.Model;
using Xunit;

public class ClavierTests
{
    [Fact]
    public void ClavierDecision_IsExactly8Bytes_Blittable()
    {
        Assert.Equal(8, Marshal.SizeOf<ClavierDecision>());
    }

    [Fact]
    public void ClavierSession_Decide_ReturnsCalibratedAction_Sub2Ms()
    {
        using var session = new ClavierSession(embeddingDim: 768, hiddenDim: 256, numActions: 16);

        string[] actions = ["BUY", "SELL", "HOLD", "CANCEL", "REBALANCE", "HEDGE"];
        string state = "Market depth: Bid 100.5 (Vol 5000), Ask 100.6 (Vol 200). Volatility expanding.";

        // Warmup JIT
        _ = session.Decide(state.AsSpan(), actions);

        var sw = Stopwatch.StartNew();
        var decision = session.Decide(state.AsSpan(), actions);
        sw.Stop();

        Assert.True(decision.ActionId < (uint)actions.Length);
        Assert.InRange(decision.Confidence, 0.0f, 1.0f);
        Assert.True(sw.Elapsed.TotalMilliseconds < 2.5, $"Decide turnaround must be < 2.5ms, took {sw.Elapsed.TotalMilliseconds:F3}ms");
    }

    [Fact]
    public void ClavierSession_DecideWithDistribution_ProducesCalibratedProbabilities()
    {
        using var session = new ClavierSession(embeddingDim: 768, hiddenDim: 256, numActions: 8, temperature: 1.2f);

        string[] actions = ["NORTH", "SOUTH", "EAST", "WEST", "UP", "DOWN", "WAIT", "INTERACT"];
        string state = "Enemy agent detected at coordinates (12, 45). Ammo low, shield at 85%.";

        var result = session.DecideWithDistribution(state.AsSpan(), actions);

        Assert.NotNull(result);
        Assert.Equal(actions.Length, result.Probabilities.Length);
        Assert.NotNull(result.ActionName);

        // Softmax probabilities must sum to 1.0
        float sumProb = 0f;
        float maxProb = 0f;
        int maxIdx = -1;
        for (int i = 0; i < result.Probabilities.Length; i++)
        {
            sumProb += result.Probabilities[i];
            if (result.Probabilities[i] > maxProb)
            {
                maxProb = result.Probabilities[i];
                maxIdx = i;
            }
        }

        Assert.True(MathF.Abs(sumProb - 1.0f) < 1e-4f, $"Probabilities must sum to 1.0, got {sumProb}");
        Assert.Equal((uint)maxIdx, result.Decision.ActionId);
        Assert.Equal(maxProb, result.Decision.Confidence, 4);
    }

    [Fact]
    public void ClavierSession_DecideBatch_MatchesIndividualDecisions()
    {
        using var session = new ClavierSession(embeddingDim: 768, hiddenDim: 256, numActions: 10);

        string[] actions = ["ACT_0", "ACT_1", "ACT_2", "ACT_3", "ACT_4"];
        ReadOnlyMemory<char>[] states =
        [
            "Player Health: 100, Stamina: 90".AsMemory(),
            "Player Health: 20, Stamina: 10".AsMemory(),
            "Player Health: 50, Stamina: 50".AsMemory()
        ];

        Span<ClavierDecision> batchDecisions = stackalloc ClavierDecision[states.Length];
        session.DecideBatch(states, batchDecisions, actions);

        for (int i = 0; i < states.Length; i++)
        {
            var single = session.Decide(states[i].Span, actions);
            Assert.Equal(single.ActionId, batchDecisions[i].ActionId);
            Assert.Equal(single.Confidence, batchDecisions[i].Confidence, 4);
        }
    }

    [Fact]
    public void BrierScoreLoss_ComputesZero_OnPerfectPredictions()
    {
        float[] probabilities = [0.0f, 1.0f, 0.0f];
        float[] targets = [0.0f, 1.0f, 0.0f];

        float loss = ClavierCalibration.BrierScore(probabilities, targets);
        Assert.True(loss < 1e-5f, $"Perfect prediction Brier score must be near 0, got {loss}");
    }

    [Fact]
    public void CategoricalCrossEntropy_LabelSmoothing_SmoothsDistribution()
    {
        float[] logits = [0f, 0f, 0f, 0f];
        float[] target = [1f, 0f, 0f, 0f];

        float lossUnsmoothed = ClavierCalibration.CategoricalCrossEntropy(logits, target, labelSmoothing: 0.0f);
        float lossSmoothed = ClavierCalibration.CategoricalCrossEntropy(logits, target, labelSmoothing: 0.1f);

        // When logits are all 0 (uniform), cross-entropy equals log(4) regardless of smoothing
        Assert.True(MathF.Abs(lossUnsmoothed - MathF.Log(4f)) < 1e-4f);
        Assert.True(MathF.Abs(lossSmoothed - MathF.Log(4f)) < 1e-4f);
    }

    [Fact]
    public void ExpectedCalibrationError_ComputesAccurately()
    {
        float[] confidences = [0.9f, 0.9f, 0.9f, 0.9f];
        int[] predictions = [1, 1, 1, 0]; // 3 correct out of 4 -> accuracy = 0.75
        int[] groundTruth = [1, 1, 1, 1];

        float ece = ClavierCalibration.ExpectedCalibrationError(confidences, predictions, groundTruth, numBins: 10);
        // Avg conf in bin 9 is 0.9, avg acc is 0.75 -> gap = |0.75 - 0.90| = 0.15
        Assert.True(MathF.Abs(ece - 0.15f) < 1e-4f, $"ECE should be ~0.15, got {ece}");
    }

    [Fact]
    public void ClavierNoul_IsExactly8Bytes_Blittable()
    {
        Assert.Equal(8, Marshal.SizeOf<ClavierNoul>());
    }

    [Fact]
    public void ClavierScore_IsExactly8Bytes_Blittable()
    {
        Assert.Equal(8, Marshal.SizeOf<ClavierScore>());
    }

    [Fact]
    public void ClavierSession_Verify_ReturnsCalibratedNoulProbability()
    {
        using var session = new ClavierSession(embeddingDim: 768, hiddenDim: 256, temperature: 1.0f);

        string state = "Database query latency spikes to 850ms, memory utilization at 96%, thread pool exhausted.";
        string hypothesis = "System is experiencing imminent resource exhaustion failure.";

        // Warmup JIT
        _ = session.VerifyWithDetails(state.AsSpan(), hypothesis.AsSpan(), threshold: 0.5f);

        var res = session.VerifyWithDetails(state.AsSpan(), hypothesis.AsSpan(), threshold: 0.5f);

        Assert.InRange(res.Probability, 0.0f, 1.0f);
        Assert.InRange(res.Confidence, 0.0f, 1.0f);
        Assert.Equal(res.Probability >= 0.5f, res.IsAffirmative);
        Assert.True(res.LatencyMs < 5.0, $"Noul turnaround must be < 5.0ms on CI, took {res.LatencyMs:F3}ms");
    }

    [Fact]
    public void ClavierSession_Score_ReturnsBoundedCalibratedScore()
    {
        using var session = new ClavierSession(embeddingDim: 768, hiddenDim: 256, temperature: 1.0f);

        string state = "Autonomous rover telemetry: incline 38 deg, wheel slip 45%, battery at 12%.";
        string criteria = "Terrain traversal hazard level";

        // Warmup JIT
        _ = session.ScoreWithDetails(state.AsSpan(), criteria.AsSpan(), min: 0.0f, max: 100.0f);

        var res = session.ScoreWithDetails(state.AsSpan(), criteria.AsSpan(), min: 0.0f, max: 100.0f);

        Assert.InRange(res.Value, 0.0f, 100.0f);
        Assert.InRange(res.Confidence, 0.0f, 1.0f);
        Assert.True(res.LatencyMs < 5.0, $"Score turnaround must be < 5.0ms on CI, took {res.LatencyMs:F3}ms");
    }

    [Fact]
    public void ClavierSession_VerifyBatch_MatchesIndividualResults()
    {
        using var session = new ClavierSession(embeddingDim: 768, hiddenDim: 256);

        ReadOnlyMemory<char>[] states =
        [
            "User provided valid signed JWT token".AsMemory(),
            "SQL syntax error near SELECT * FROM users WHERE 1=1; DROP TABLE users;".AsMemory(),
            "Request rate: 5 requests per minute from trusted internal IP".AsMemory()
        ];
        string hypothesis = "Request represents a potential security attack";

        Span<ClavierNoul> batchResults = stackalloc ClavierNoul[states.Length];
        session.VerifyBatch(states, batchResults, hypothesis.AsSpan(), threshold: 0.5f);

        for (int i = 0; i < states.Length; i++)
        {
            var single = session.Verify(states[i].Span, hypothesis.AsSpan(), threshold: 0.5f);
            Assert.Equal(single.IsAffirmative, batchResults[i].IsAffirmative);
            Assert.Equal(single.Probability, batchResults[i].Probability, 4);
        }
    }

    [Fact]
    public void ClavierSession_ScoreBatch_MatchesIndividualResults()
    {
        using var session = new ClavierSession(embeddingDim: 768, hiddenDim: 256);

        ReadOnlyMemory<char>[] states =
        [
            "Customer sentiment: 'Great service, loved the product!'".AsMemory(),
            "Customer sentiment: 'Item arrived broken, unacceptable delay'".AsMemory(),
            "Customer sentiment: 'Package delivered on time'".AsMemory()
        ];
        string criteria = "Customer satisfaction rating";

        Span<ClavierScore> batchScores = stackalloc ClavierScore[states.Length];
        session.ScoreBatch(states, batchScores, criteria.AsSpan(), min: 1.0f, max: 5.0f);

        for (int i = 0; i < states.Length; i++)
        {
            var single = session.Score(states[i].Span, criteria.AsSpan(), min: 1.0f, max: 5.0f);
            Assert.Equal(single.Value, batchScores[i].Value, 3);
            Assert.Equal(single.Confidence, batchScores[i].Confidence, 4);
        }
    }
}
