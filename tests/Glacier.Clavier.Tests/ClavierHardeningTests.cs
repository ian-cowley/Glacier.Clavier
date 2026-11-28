namespace Glacier.Clavier.Tests;

using System;
using System.Buffers.Binary;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Glacier.Clavier.Engine;
using Glacier.Clavier.Model;
using Xunit;

public class ClavierHardeningTests
{
    [Theory]
    [InlineData(0.5f, 0.0f)]
    [InlineData(1.0f, 0.0f)]
    [InlineData(2.5f, 0.0f)]
    [InlineData(1.0f, 30.0f)]
    [InlineData(1.0f, -30.0f)]
    public void ClavierScoreHead_TrainStep_BceLossMatchesFiniteDifferenceGradient(float temperature, float biasShift)
    {
        const int inputDim = 16;
        const int hiddenDim = 8;
        var baseHead = new ClavierScoreHead(inputDim, hiddenDim, temperature);

        byte[] baseBytes;
        using (var ms = new MemoryStream())
        {
            baseHead.SaveWeights(ms);
            baseBytes = ms.ToArray();
        }

        // Layout: magic(4) + inDim(4) + hidDim(4) + temp(4) + w1(hidden*in*4) + b1(hidden*4) + wVal(hidden*4) + bVal(4) + wConf(hidden*4) + bConf(4)
        int bValOffset = 16 + (hiddenDim * inputDim + hiddenDim + hiddenDim) * 4;
        float origBVal = BinaryPrimitives.ReadSingleLittleEndian(baseBytes.AsSpan(bValOffset, 4)) + biasShift;
        BinaryPrimitives.WriteSingleLittleEndian(baseBytes.AsSpan(bValOffset, 4), origBVal);

        var state = new float[inputDim];
        for (int i = 0; i < inputDim; i++) state[i] = MathF.Sin(i * 0.37f) * 0.5f;

        const float min = 0.0f;
        const float max = 100.0f;
        const float target = 78.5f;
        const float lr = 0.05f;
        const float eps = 1e-2f;

        var headPlus = new ClavierScoreHead(inputDim, hiddenDim, temperature);
        var headMinus = new ClavierScoreHead(inputDim, hiddenDim, temperature);
        var headCenter = new ClavierScoreHead(inputDim, hiddenDim, temperature);

        byte[] plusBytes = (byte[])baseBytes.Clone();
        BinaryPrimitives.WriteSingleLittleEndian(plusBytes.AsSpan(bValOffset, 4), origBVal + eps);
        byte[] minusBytes = (byte[])baseBytes.Clone();
        BinaryPrimitives.WriteSingleLittleEndian(minusBytes.AsSpan(bValOffset, 4), origBVal - eps);

        headPlus.LoadWeights(new MemoryStream(plusBytes));
        headMinus.LoadWeights(new MemoryStream(minusBytes));
        headCenter.LoadWeights(new MemoryStream(baseBytes));

        float lossPlus = headPlus.TrainStep(state, target, min, max, lr);
        float lossMinus = headMinus.TrainStep(state, target, min, max, lr);
        float lossCenter = headCenter.TrainStep(state, target, min, max, lr);

        Assert.True(float.IsFinite(lossPlus) && lossPlus >= 0f);
        Assert.True(float.IsFinite(lossMinus) && lossMinus >= 0f);
        Assert.True(float.IsFinite(lossCenter) && lossCenter >= 0f);

        using var afterMs = new MemoryStream();
        headCenter.SaveWeights(afterMs);
        float updatedBVal = BinaryPrimitives.ReadSingleLittleEndian(afterMs.ToArray().AsSpan(bValOffset, 4));

        float analyticGrad = (origBVal - updatedBVal) / lr;
        float numericalGrad = (lossPlus - lossMinus) / (2.0f * eps);

        Assert.Equal(numericalGrad, analyticGrad, tolerance: 2e-2f);
    }

    [Fact]
    public void ClavierScoreHead_ExtremeFiniteMinMax_AvoidsDoubleOverflow()
    {
        var head = new ClavierScoreHead(inputDim: 8, hiddenDim: 4, temperature: 1.0f);
        float[] state = [0.2f, -0.3f, 0.5f, 0.1f, -0.4f, 0.6f, -0.1f, 0.2f];
        float[] hidden = new float[4];

        float min = -3e38f;
        float max = 3e38f;

        var score = head.Forward(state, hidden, min, max);
        Assert.True(float.IsFinite(score.Value));
        Assert.InRange(score.Value, min, max);

        float loss = head.TrainStep(state, targetValue: 1e38f, min: min, max: max, learningRate: 0.05f);
        Assert.True(float.IsFinite(loss) && loss >= 0f);
    }

    [Fact]
    public void ClavierSession_Temperature_LocksValidatesAndChecksDisposed()
    {
        using var session = new ClavierSession(embeddingDim: 32, hiddenDim: 16, numActions: 4, temperature: 1.0f);
        Assert.Equal(1.0f, session.Temperature);

        Assert.Throws<ArgumentOutOfRangeException>(() => session.Temperature = 0f);
        Assert.Throws<ArgumentOutOfRangeException>(() => session.Temperature = -1.5f);
        Assert.Throws<ArgumentOutOfRangeException>(() => session.Temperature = float.NaN);
        Assert.Throws<ArgumentOutOfRangeException>(() => session.Temperature = float.PositiveInfinity);
        Assert.Throws<ArgumentOutOfRangeException>(() => session.Temperature = float.NegativeInfinity);

        session.Temperature = 2.5f;
        Assert.Equal(2.5f, session.Temperature);

        using var ms = new MemoryStream();
        session.SaveCheckpoint(ms);
        ms.Position = 0;

        using var restored = new ClavierSession(embeddingDim: 32, hiddenDim: 16, numActions: 4, temperature: 1.0f);
        restored.LoadCheckpoint(ms);
        Assert.Equal(2.5f, restored.Temperature);

        // Verify all 3 heads received identical temperature
        Assert.Equal(
            session.Verify("test state".AsSpan(), "ok".AsSpan()).Probability,
            restored.Verify("test state".AsSpan(), "ok".AsSpan()).Probability,
            precision: 6);
        Assert.Equal(
            session.Score("test state".AsSpan(), "quality".AsSpan()).Value,
            restored.Score("test state".AsSpan(), "quality".AsSpan()).Value,
            precision: 5);

        session.Dispose();
        Assert.Throws<ObjectDisposedException>(() => _ = session.Temperature);
        Assert.Throws<ObjectDisposedException>(() => session.Temperature = 1.0f);
    }

    [Fact]
    public void ClavierSession_ConcurrentReadersAndWriters_UnderRwLock_AreThreadSafe()
    {
        using var session = new ClavierSession(embeddingDim: 32, hiddenDim: 16, numActions: 4);
        var states = new ReadOnlyMemory<char>[20];
        for (int i = 0; i < states.Length; i++) states[i] = $"state_{i}".AsMemory();
        string[] actions = ["A0", "A1", "A2"];

        Parallel.For(0, 40, i =>
        {
            if (i % 5 == 0)
            {
                session.Temperature = 0.5f + (i % 4) * 0.5f;
                session.TrainDecision("state_0".AsSpan(), i % 3, actions, learningRate: 0.01f, labelSmoothing: 0.1f);
                session.TrainVerify("state_0".AsSpan(), "hyp".AsSpan(), i % 2 == 0, learningRate: 0.01f);
                session.TrainScore("state_0".AsSpan(), "crit".AsSpan(), 0.5f, 0f, 1f, learningRate: 0.01f);
            }
            else
            {
                float temp = session.Temperature;
                Assert.True(float.IsFinite(temp) && temp > 0f);

                var decs = new ClavierDecision[states.Length];
                var nouls = new ClavierNoul[states.Length];
                var scores = new ClavierScore[states.Length];

                session.DecideBatch(states, decs, actions);
                session.VerifyBatch(states, nouls, "hyp".AsSpan(), 0.5f);
                session.ScoreBatch(states, scores, "crit".AsSpan(), 0f, 1f);

                for (int j = 0; j < states.Length; j++)
                {
                    Assert.True(float.IsFinite(decs[j].Confidence));
                    Assert.True(float.IsFinite(nouls[j].Probability));
                    Assert.True(float.IsFinite(scores[j].Value));
                }
            }
        });
    }

    [Fact]
    public void DecideConformalSet_PreservesExplicitQuantileAfterCalibrationAndCheckpointReload()
    {
        using var session = new ClavierSession(embeddingDim: 32, hiddenDim: 16, numActions: 4);
        float[] scores = [0.05f, 0.10f, 0.15f, 0.20f, 0.25f, 0.30f, 0.35f, 0.40f, 0.45f, 0.50f,
                          0.55f, 0.60f, 0.65f, 0.70f, 0.75f, 0.80f, 0.85f, 0.90f, 0.92f, 0.94f];
        session.CalibrateConformal(scores, alpha: 0.05f);

        // Explicit override after calibration must win even when alpha != _conformalAlpha
        session.ConformalQuantile = 0.80f;
        var setBeforeSave = session.DecideConformalSet("deploy service".AsSpan(), alpha: 0.20f);
        Assert.Equal(0.80f, setBeforeSave.QuantileThreshold, precision: 5);

        // Save V2 checkpoint and reload into fresh session
        using var ms = new MemoryStream();
        session.SaveCheckpoint(ms);
        ms.Position = 0;

        using var restored = new ClavierSession(embeddingDim: 32, hiddenDim: 16, numActions: 4);
        restored.LoadCheckpoint(ms);
        var setAfterLoad = restored.DecideConformalSet("deploy service".AsSpan(), alpha: 0.20f);
        Assert.Equal(0.80f, setAfterLoad.QuantileThreshold, precision: 5);

        // Uncalibrated session with explicit quantile also preserves across V2 checkpoint reload
        using var uncal = new ClavierSession(embeddingDim: 32, hiddenDim: 16, numActions: 4);
        uncal.ConformalQuantile = 0.77f;
        using var msUncal = new MemoryStream();
        uncal.SaveCheckpoint(msUncal);
        msUncal.Position = 0;

        using var uncalRestored = new ClavierSession(embeddingDim: 32, hiddenDim: 16, numActions: 4);
        uncalRestored.LoadCheckpoint(msUncal);
        var uncalSet = uncalRestored.DecideConformalSet("deploy service".AsSpan(), alpha: 0.40f);
        Assert.Equal(0.77f, uncalSet.QuantileThreshold, precision: 5);
    }

    [Fact]
    public void LoadCheckpoint_V1ResetsExplicitQuantileFlag_AndCalibrateConformalIsAtomicOnError()
    {
        using var session = new ClavierSession(embeddingDim: 32, hiddenDim: 16, numActions: 4);
        session.ConformalQuantile = 0.88f;
        session.ConformalAlpha = 0.12f;

        // Invalid CalibrateConformal must leave existing state untouched
        Assert.Throws<ArgumentException>(() => session.CalibrateConformal([0.1f, float.NaN], alpha: 0.05f));
        Assert.Equal(0.88f, session.ConformalQuantile, precision: 5);
        Assert.Equal(0.12f, session.ConformalAlpha, precision: 5);

        // Synthesize V1 checkpoint
        using var ms = new MemoryStream();
        using (var writer = new System.IO.BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            writer.Write(0x52564C43u);
            writer.Write(1);
            writer.Write(32);
            writer.Write(16);
            writer.Write(4);
            using var dh = new ClavierDecisionHead(32, 16, 4);
            var nh = new ClavierNoulHead(32, 16);
            var sh = new ClavierScoreHead(32, 16);
            dh.WriteWeights(writer);
            nh.WriteWeights(writer);
            sh.WriteWeights(writer);
        }
        ms.Position = 0;

        session.LoadCheckpoint(ms);
        // Since _hasExplicitQuantile was reset to false by V1 load, alpha != 0.05 derives 1 - alpha
        var set = session.DecideConformalSet("query".AsSpan(), alpha: 0.30f);
        Assert.Equal(0.70f, set.QuantileThreshold, precision: 5);
    }

    [Fact]
    public void TrainDecision_DynamicCandidates_HonorsLabelSmoothingAndRejectsInvalidArgs()
    {
        string[] candidates = ["ACT_0", "ACT_1", "ACT_2"];

        using var sUnsmoothed = new ClavierSession(embeddingDim: 32, hiddenDim: 16, numActions: 4);
        using var sSmoothed = new ClavierSession(embeddingDim: 32, hiddenDim: 16, numActions: 4);

        float loss0 = sUnsmoothed.TrainDecision("state".AsSpan(), 1, candidates, learningRate: 0.05f, labelSmoothing: 0.0f);
        float lossSmooth = sSmoothed.TrainDecision("state".AsSpan(), 1, candidates, learningRate: 0.05f, labelSmoothing: 0.3f);

        Assert.NotEqual(loss0, lossSmooth);

        // After smoothing update, predicted confidence for target should be lower than unsmoothed after multiple steps
        for (int i = 0; i < 15; i++)
        {
            sUnsmoothed.TrainDecision("state".AsSpan(), 1, candidates, learningRate: 0.1f, labelSmoothing: 0.0f);
            sSmoothed.TrainDecision("state".AsSpan(), 1, candidates, learningRate: 0.1f, labelSmoothing: 0.3f);
        }
        var p0 = sUnsmoothed.DecideWithDistribution("state".AsSpan(), candidates).Probabilities[1];
        var pSmooth = sSmoothed.DecideWithDistribution("state".AsSpan(), candidates).Probabilities[1];
        Assert.True(p0 > pSmooth, $"Expected unsmoothed prob ({p0}) > smoothed prob ({pSmooth})");

        Assert.Throws<ArgumentOutOfRangeException>(() => sSmoothed.TrainDecision("state".AsSpan(), 1, candidates, learningRate: 0.05f, labelSmoothing: -0.01f));
        Assert.Throws<ArgumentOutOfRangeException>(() => sSmoothed.TrainDecision("state".AsSpan(), 1, candidates, learningRate: 0.05f, labelSmoothing: 1.01f));
        Assert.Throws<ArgumentOutOfRangeException>(() => sSmoothed.TrainDecision("state".AsSpan(), 1, candidates, learningRate: 0.05f, labelSmoothing: float.NaN));
        Assert.Throws<ArgumentOutOfRangeException>(() => sSmoothed.TrainDecision("state".AsSpan(), 1, candidates, learningRate: float.PositiveInfinity, labelSmoothing: 0.0f));
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    [InlineData(0.0f)]
    [InlineData(-1.0f)]
    public void Heads_InvalidTemperatureAndLearningRate_ThrowArgumentOutOfRangeException(float badVal)
    {
        // Constructor falls back to 1.0f cleanly
        using var dh = new ClavierDecisionHead(8, 4, 2, badVal);
        var nh = new ClavierNoulHead(8, 4, badVal);
        var sh = new ClavierScoreHead(8, 4, badVal);
        Assert.Equal(1.0f, dh.Temperature);
        Assert.Equal(1.0f, nh.Temperature);
        Assert.Equal(1.0f, sh.Temperature);

        // Setters reject all invalid values
        Assert.Throws<ArgumentOutOfRangeException>(() => dh.Temperature = badVal);
        Assert.Throws<ArgumentOutOfRangeException>(() => nh.Temperature = badVal);
        Assert.Throws<ArgumentOutOfRangeException>(() => sh.Temperature = badVal);

        // TrainStep learningRate rejects all invalid values
        float[] state = new float[8];
        float[] candEmb = new float[16];
        Assert.Throws<ArgumentOutOfRangeException>(() => dh.TrainStep(state, 0, badVal));
        Assert.Throws<ArgumentOutOfRangeException>(() => dh.TrainCandidateStep(candEmb, 2, 0, badVal));
        Assert.Throws<ArgumentOutOfRangeException>(() => nh.TrainStep(state, true, badVal));
        Assert.Throws<ArgumentOutOfRangeException>(() => sh.TrainStep(state, 0.5f, 0f, 1f, badVal));
    }

    [Fact]
    public void Heads_OverflowingDimensionsAndInvalidBounds_ThrowExpectedExceptions()
    {
        Assert.Throws<OverflowException>(() => new ClavierDecisionHead(65536, 65536, 4));
        Assert.Throws<OverflowException>(() => new ClavierNoulHead(65536, 65536));
        Assert.Throws<OverflowException>(() => new ClavierScoreHead(65536, 65536));

        var nh = new ClavierNoulHead(8, 4);
        var sh = new ClavierScoreHead(8, 4);
        using var dh = new ClavierDecisionHead(8, 4, 2);
        float[] state = new float[8];
        float[] hidden = new float[4];

        Assert.Throws<ArgumentOutOfRangeException>(() => nh.Forward(state, hidden, float.NaN));
        Assert.Throws<ArgumentException>(() => sh.Forward(state, hidden, float.NaN, 1f));
        Assert.Throws<ArgumentException>(() => sh.Forward(state, hidden, 0f, float.PositiveInfinity));
        Assert.Throws<ArgumentException>(() => sh.TrainStep(state, 0.5f, float.NaN, 1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => sh.TrainStep(state, float.NaN, 0f, 1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => sh.TrainStep(state, float.PositiveInfinity, 0f, 1f));

        state[3] = float.NaN;
        Assert.Throws<ArgumentException>(() => dh.TrainStep(state, 0));
        Assert.Throws<ArgumentException>(() => nh.TrainStep(state, true));
        Assert.Throws<ArgumentException>(() => sh.TrainStep(state, 0.5f, 0f, 1f));
    }

    [Fact]
    public void ClavierCalibration_InvalidInputsAndLargeBrierScore_HandledProperly()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => ClavierCalibration.ExpectedCalibrationError([0.5f], [1], [1], numBins: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => ClavierCalibration.ExpectedCalibrationError([0.5f], [1], [1], numBins: -3));
        Assert.Throws<ArgumentException>(() => ClavierCalibration.ExpectedCalibrationError([float.NaN], [1], [1], numBins: 10));
        Assert.Throws<ArgumentException>(() => ClavierCalibration.CalibrateQuantile([0.1f, float.NaN], 0.5f));
        Assert.Throws<ArgumentException>(() => ClavierCalibration.CalibrateQuantile([float.PositiveInfinity], 0.05f));
        Assert.Throws<ArgumentOutOfRangeException>(() => ClavierCalibration.ComputeNonConformityScores([], [], 0, Span<float>.Empty));
        Assert.Throws<ArgumentException>(() => ClavierCalibration.ComputeNonConformityScores([float.NaN, 0.5f], [0], 2, new float[1]));
        Assert.Throws<ArgumentException>(() => ClavierCalibration.CategoricalCrossEntropy([float.NaN, 0f], [1f, 0f]));
        Assert.Throws<ArgumentOutOfRangeException>(() => ClavierCalibration.CategoricalCrossEntropy([1f, 0f], [1f, 0f], -0.1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => ClavierCalibration.CategoricalCrossEntropy([1f, 0f], [1f, 0f], 1.1f));

        // Extreme logits remain finite via Log1p log-sum-exp
        float ce = ClavierCalibration.CategoricalCrossEntropy([1000f, -1000f], [1f, 0f]);
        Assert.True(float.IsFinite(ce) && ce >= 0f);

        // Large BrierScore > 256 exercises chunked SIMD path
        float[] probs = new float[500];
        float[] targs = new float[500];
        for (int i = 0; i < 500; i++)
        {
            probs[i] = 0.75f;
            targs[i] = 0.25f;
        }
        Assert.Equal(0.25f, ClavierCalibration.BrierScore(probs, targs), precision: 5);
        probs[499] = float.NaN;
        Assert.Throws<ArgumentException>(() => ClavierCalibration.BrierScore(probs, targs));
    }

    [Fact]
    public void Heads_BulkMemoryMarshalIO_RoundTripsAndRejectsCorruptOrTruncatedStreams()
    {
        // HiddenDim=600 (> 512) also exercises ArrayPool path in TrainCandidateStep
        using var dh = new ClavierDecisionHead(137, 600, 11, temperature: 1.3f);
        var nh = new ClavierNoulHead(137, 53, temperature: 0.8f);
        var sh = new ClavierScoreHead(137, 53, temperature: 1.7f);

        float[] state = Enumerable.Range(0, 137).Select(i => (i % 9) * 0.1f).ToArray();
        float[] hidden600 = new float[600];
        float[] hidden53 = new float[53];
        float[] logits = new float[11];
        float[] probs = new float[11];

        // Exercise TrainCandidateStep with hiddenDim > 512
        float[] joint = new float[2 * 137];
        Array.Copy(state, 0, joint, 0, 137);
        Array.Copy(state, 0, joint, 137, 137);
        float candLoss = dh.TrainCandidateStep(joint, 2, 1, learningRate: 0.01f, labelSmoothing: 0.1f);
        Assert.True(float.IsFinite(candLoss));

        // DecisionHead round-trip + corrupt bias + corrupt array NaN + truncation
        byte[] dhBytes;
        using (var ms = new MemoryStream()) { dh.SaveWeights(ms); dhBytes = ms.ToArray(); }
        using var dh2 = new ClavierDecisionHead(137, 600, 11);
        dh2.LoadWeights(new MemoryStream(dhBytes));
        Assert.Equal(dh.Forward(state, hidden600, logits, probs).Confidence, dh2.Forward(state, hidden600, logits, probs).Confidence, precision: 6);

        byte[] dhTrunc = dhBytes[..^10];
        Assert.Throws<EndOfStreamException>(() => dh2.LoadWeights(new MemoryStream(dhTrunc)));

        byte[] dhNanArray = (byte[])dhBytes.Clone();
        BinaryPrimitives.WriteSingleLittleEndian(dhNanArray.AsSpan(20, 4), float.NaN);
        Assert.Throws<InvalidDataException>(() => dh2.LoadWeights(new MemoryStream(dhNanArray)));

        byte[] dhNanBias = (byte[])dhBytes.Clone();
        BinaryPrimitives.WriteSingleLittleEndian(dhNanBias.AsSpan(dhNanBias.Length - 4, 4), float.PositiveInfinity);
        Assert.Throws<InvalidDataException>(() => dh2.LoadWeights(new MemoryStream(dhNanBias)));

        // NoulHead round-trip + corrupt temp + corrupt bias + truncation
        byte[] nhBytes;
        using (var ms = new MemoryStream()) { nh.SaveWeights(ms); nhBytes = ms.ToArray(); }
        var nh2 = new ClavierNoulHead(137, 53);
        nh2.LoadWeights(new MemoryStream(nhBytes));
        Assert.Equal(nh.Forward(state, hidden53).Probability, nh2.Forward(state, hidden53).Probability, precision: 6);
        Assert.Throws<EndOfStreamException>(() => nh2.LoadWeights(new MemoryStream(nhBytes[..^10])));

        byte[] nhBadTemp = (byte[])nhBytes.Clone();
        BinaryPrimitives.WriteSingleLittleEndian(nhBadTemp.AsSpan(12, 4), float.NaN);
        Assert.Throws<InvalidDataException>(() => nh2.LoadWeights(new MemoryStream(nhBadTemp)));

        byte[] nhNanBias = (byte[])nhBytes.Clone();
        BinaryPrimitives.WriteSingleLittleEndian(nhNanBias.AsSpan(nhNanBias.Length - 4, 4), float.NaN);
        Assert.Throws<InvalidDataException>(() => nh2.LoadWeights(new MemoryStream(nhNanBias)));

        // ScoreHead round-trip + corrupt temp + corrupt bias + truncation
        byte[] shBytes;
        using (var ms = new MemoryStream()) { sh.SaveWeights(ms); shBytes = ms.ToArray(); }
        var sh2 = new ClavierScoreHead(137, 53);
        sh2.LoadWeights(new MemoryStream(shBytes));
        Assert.Equal(sh.Forward(state, hidden53).Value, sh2.Forward(state, hidden53).Value, precision: 6);
        Assert.Throws<EndOfStreamException>(() => sh2.LoadWeights(new MemoryStream(shBytes[..^10])));

        byte[] shBadTemp = (byte[])shBytes.Clone();
        BinaryPrimitives.WriteSingleLittleEndian(shBadTemp.AsSpan(12, 4), -1.0f);
        Assert.Throws<InvalidDataException>(() => sh2.LoadWeights(new MemoryStream(shBadTemp)));

        byte[] shNanConfBias = (byte[])shBytes.Clone();
        BinaryPrimitives.WriteSingleLittleEndian(shNanConfBias.AsSpan(shNanConfBias.Length - 4, 4), float.NegativeInfinity);
        Assert.Throws<InvalidDataException>(() => sh2.LoadWeights(new MemoryStream(shNanConfBias)));
    }

    [Fact]
    public void ClavierSession_BatchInferenceAboveThreshold16_VaryingCandidateAndTextLengths_NoArrayPoolTaint()
    {
        using var session = new ClavierSession(embeddingDim: 64, hiddenDim: 32, numActions: 6);
        const int count = 24;
        var states = new ReadOnlyMemory<char>[count];
        for (int i = 0; i < count; i++) states[i] = $"Observation #{i}: latency={i * 3}ms".AsMemory();

        string[] cand5 = ["ACT_A", "ACT_B", "ACT_C", "ACT_D", "ACT_E"];
        string[] cand2 = ["YES", "NO"];

        var decOut = new ClavierDecision[count];
        var noulOut = new ClavierNoul[count];
        var scoreOut = new ClavierScore[count];

        // 1. DecideBatch with 5 candidates, then 2 candidates, then empty candidates
        session.DecideBatch(states, decOut, cand5);
        for (int i = 0; i < count; i++)
        {
            var single = session.Decide(states[i].Span, cand5);
            Assert.Equal(single.ActionId, decOut[i].ActionId);
            Assert.Equal(single.Confidence, decOut[i].Confidence, precision: 5);
        }

        session.DecideBatch(states, decOut, cand2);
        for (int i = 0; i < count; i++)
        {
            var single = session.Decide(states[i].Span, cand2);
            Assert.Equal(single.ActionId, decOut[i].ActionId);
            Assert.Equal(single.Confidence, decOut[i].Confidence, precision: 5);
        }

        session.DecideBatch(states, decOut);
        for (int i = 0; i < count; i++)
        {
            var single = session.Decide(states[i].Span);
            Assert.Equal(single.ActionId, decOut[i].ActionId);
            Assert.Equal(single.Confidence, decOut[i].Confidence, precision: 5);
        }

        // 2. VerifyBatch with long hypothesis, then short hypothesis, then empty
        session.VerifyBatch(states, noulOut, "A very long hypothesis string exceeding short pool length".AsSpan(), 0.5f);
        session.VerifyBatch(states, noulOut, "ok".AsSpan(), 0.5f);
        for (int i = 0; i < count; i++)
        {
            var single = session.Verify(states[i].Span, "ok".AsSpan(), 0.5f);
            Assert.Equal(single.Probability, noulOut[i].Probability, precision: 5);
        }

        // 3. ScoreBatch with long criteria, then short criteria
        session.ScoreBatch(states, scoreOut, "Long multi-word evaluation criteria string".AsSpan(), 0f, 10f);
        session.ScoreBatch(states, scoreOut, "q".AsSpan(), 0f, 10f);
        for (int i = 0; i < count; i++)
        {
            var single = session.Score(states[i].Span, "q".AsSpan(), 0f, 10f);
            Assert.Equal(single.Value, scoreOut[i].Value, precision: 4);
        }
    }

    private static int FindSubsequence(byte[] haystack, uint magic)
    {
        Span<byte> needle = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(needle, magic);
        return haystack.AsSpan().IndexOf(needle);
    }

    [Fact]
    public void ClavierSession_LoadCheckpoint_AtomicRollbackOnMidReadFailure()
    {
        using var sessionA = new ClavierSession(embeddingDim: 32, hiddenDim: 16, numActions: 4, temperature: 1.0f);
        sessionA.ConformalAlpha = 0.12f;
        sessionA.ConformalQuantile = 0.82f;

        var origDec = sessionA.Decide("state_probe".AsSpan());
        var origNoul = sessionA.Verify("state_probe".AsSpan(), "hyp".AsSpan());
        var origScore = sessionA.Score("state_probe".AsSpan(), "crit".AsSpan());

        using var sessionB = new ClavierSession(embeddingDim: 32, hiddenDim: 16, numActions: 4, temperature: 3.5f);
        for (int i = 0; i < 10; i++)
        {
            sessionB.TrainDecision("state_probe".AsSpan(), 2, learningRate: 0.1f);
            sessionB.TrainVerify("state_probe".AsSpan(), "hyp".AsSpan(), true, learningRate: 0.1f);
            sessionB.TrainScore("state_probe".AsSpan(), "crit".AsSpan(), 0.9f, 0f, 1f, learningRate: 0.1f);
        }
        sessionB.CalibrateConformal([0.1f, 0.3f, 0.6f, 0.9f], alpha: 0.20f);

        byte[] ckptB;
        using (var ms = new MemoryStream())
        {
            sessionB.SaveCheckpoint(ms);
            ckptB = ms.ToArray();
        }

        // 1. Corrupt NoulHead magic ("CLNH" = 0x484E4C43u)
        int noulOffset = FindSubsequence(ckptB, 0x484E4C43u);
        Assert.True(noulOffset > 0);
        byte[] badNoulMagic = (byte[])ckptB.Clone();
        badNoulMagic[noulOffset] ^= 0xFF;

        Assert.Throws<InvalidDataException>(() => sessionA.LoadCheckpoint(new MemoryStream(badNoulMagic)));
        Assert.Equal(1.0f, sessionA.Temperature);
        Assert.Equal(0.82f, sessionA.ConformalQuantile);
        Assert.Equal(0.12f, sessionA.ConformalAlpha);
        Assert.Equal(origDec, sessionA.Decide("state_probe".AsSpan()));
        Assert.Equal(origNoul, sessionA.Verify("state_probe".AsSpan(), "hyp".AsSpan()));
        Assert.Equal(origScore, sessionA.Score("state_probe".AsSpan(), "crit".AsSpan()));

        // 2. Corrupt ScoreHead magic ("CLSH" = 0x48534C43u)
        int scoreOffset = FindSubsequence(ckptB, 0x48534C43u);
        Assert.True(scoreOffset > noulOffset);
        byte[] badScoreMagic = (byte[])ckptB.Clone();
        badScoreMagic[scoreOffset] ^= 0xFF;

        Assert.Throws<InvalidDataException>(() => sessionA.LoadCheckpoint(new MemoryStream(badScoreMagic)));
        Assert.Equal(1.0f, sessionA.Temperature);
        Assert.Equal(origDec, sessionA.Decide("state_probe".AsSpan()));
        Assert.Equal(origNoul, sessionA.Verify("state_probe".AsSpan(), "hyp".AsSpan()));
        Assert.Equal(origScore, sessionA.Score("state_probe".AsSpan(), "crit".AsSpan()));

        // 3. Truncate inside ScoreHead
        Assert.Throws<EndOfStreamException>(() => sessionA.LoadCheckpoint(new MemoryStream(ckptB[..^8])));
        Assert.Equal(1.0f, sessionA.Temperature);
        Assert.Equal(origDec, sessionA.Decide("state_probe".AsSpan()));
        Assert.Equal(origNoul, sessionA.Verify("state_probe".AsSpan(), "hyp".AsSpan()));
        Assert.Equal(origScore, sessionA.Score("state_probe".AsSpan(), "crit".AsSpan()));
    }

    [Fact]
    public void Heads_And_Session_LoadWithNanOrTruncation_DoNotPoisonLiveWeights()
    {
        float[] state = Enumerable.Range(0, 16).Select(i => (i + 1) * 0.05f).ToArray();
        float[] hidden = new float[8];
        float[] logits = new float[4];
        float[] probs = new float[4];

        // 1. ClavierDecisionHead atomic rollback on NaN in _w1, NaN in _bCand, and truncation
        using var dh = new ClavierDecisionHead(16, 8, 4, temperature: 1.0f);
        var origDhOut = dh.Forward(state, hidden, logits, probs);

        using var dhSrc = new ClavierDecisionHead(16, 8, 4, temperature: 2.5f);
        dhSrc.TrainStep(state, 2, learningRate: 0.2f);
        byte[] dhBytes;
        using (var ms = new MemoryStream()) { dhSrc.SaveWeights(ms); dhBytes = ms.ToArray(); }

        byte[] dhNanW1 = (byte[])dhBytes.Clone();
        BinaryPrimitives.WriteSingleLittleEndian(dhNanW1.AsSpan(20, 4), float.NaN);
        Assert.Throws<InvalidDataException>(() => dh.LoadWeights(new MemoryStream(dhNanW1)));
        Assert.Equal(1.0f, dh.Temperature);
        Assert.Equal(origDhOut, dh.Forward(state, hidden, logits, probs));

        byte[] dhNanBCand = (byte[])dhBytes.Clone();
        BinaryPrimitives.WriteSingleLittleEndian(dhNanBCand.AsSpan(dhNanBCand.Length - 4, 4), float.NaN);
        Assert.Throws<InvalidDataException>(() => dh.LoadWeights(new MemoryStream(dhNanBCand)));
        Assert.Equal(1.0f, dh.Temperature);
        Assert.Equal(origDhOut, dh.Forward(state, hidden, logits, probs));

        Assert.Throws<EndOfStreamException>(() => dh.LoadWeights(new MemoryStream(dhBytes[..^4])));
        Assert.Equal(1.0f, dh.Temperature);
        Assert.Equal(origDhOut, dh.Forward(state, hidden, logits, probs));

        // 2. ClavierNoulHead atomic rollback on NaN in _w1, _b2, and truncation
        var nh = new ClavierNoulHead(16, 8, temperature: 1.0f);
        var origNhOut = nh.Forward(state, hidden);

        var nhSrc = new ClavierNoulHead(16, 8, temperature: 2.5f);
        nhSrc.TrainStep(state, true, learningRate: 0.2f);
        byte[] nhBytes;
        using (var ms = new MemoryStream()) { nhSrc.SaveWeights(ms); nhBytes = ms.ToArray(); }

        byte[] nhNanW1 = (byte[])nhBytes.Clone();
        BinaryPrimitives.WriteSingleLittleEndian(nhNanW1.AsSpan(16, 4), float.NaN);
        Assert.Throws<InvalidDataException>(() => nh.LoadWeights(new MemoryStream(nhNanW1)));
        Assert.Equal(1.0f, nh.Temperature);
        Assert.Equal(origNhOut, nh.Forward(state, hidden));

        byte[] nhNanB2 = (byte[])nhBytes.Clone();
        BinaryPrimitives.WriteSingleLittleEndian(nhNanB2.AsSpan(nhNanB2.Length - 4, 4), float.NaN);
        Assert.Throws<InvalidDataException>(() => nh.LoadWeights(new MemoryStream(nhNanB2)));
        Assert.Equal(1.0f, nh.Temperature);
        Assert.Equal(origNhOut, nh.Forward(state, hidden));

        Assert.Throws<EndOfStreamException>(() => nh.LoadWeights(new MemoryStream(nhBytes[..^4])));
        Assert.Equal(1.0f, nh.Temperature);
        Assert.Equal(origNhOut, nh.Forward(state, hidden));

        // 3. ClavierScoreHead atomic rollback on NaN in _w1, _wConf, _bConf, and truncation
        var sh = new ClavierScoreHead(16, 8, temperature: 1.0f);
        var origShOut = sh.Forward(state, hidden);

        var shSrc = new ClavierScoreHead(16, 8, temperature: 2.5f);
        shSrc.TrainStep(state, 0.85f, 0f, 1f, learningRate: 0.2f);
        byte[] shBytes;
        using (var ms = new MemoryStream()) { shSrc.SaveWeights(ms); shBytes = ms.ToArray(); }

        byte[] shNanW1 = (byte[])shBytes.Clone();
        BinaryPrimitives.WriteSingleLittleEndian(shNanW1.AsSpan(16, 4), float.NaN);
        Assert.Throws<InvalidDataException>(() => sh.LoadWeights(new MemoryStream(shNanW1)));
        Assert.Equal(1.0f, sh.Temperature);
        Assert.Equal(origShOut, sh.Forward(state, hidden));

        byte[] shNanConf = (byte[])shBytes.Clone();
        BinaryPrimitives.WriteSingleLittleEndian(shNanConf.AsSpan(shNanConf.Length - 4, 4), float.PositiveInfinity);
        Assert.Throws<InvalidDataException>(() => sh.LoadWeights(new MemoryStream(shNanConf)));
        Assert.Equal(1.0f, sh.Temperature);
        Assert.Equal(origShOut, sh.Forward(state, hidden));

        Assert.Throws<EndOfStreamException>(() => sh.LoadWeights(new MemoryStream(shBytes[..^4])));
        Assert.Equal(1.0f, sh.Temperature);
        Assert.Equal(origShOut, sh.Forward(state, hidden));

        // 4. ClavierSession.LoadCheckpoint atomic rollback when DecisionHead._w1 contains NaN
        using var session = new ClavierSession(embeddingDim: 32, hiddenDim: 16, numActions: 4, temperature: 1.0f);
        var origSessDec = session.Decide("state_probe".AsSpan());

        using var sessionSrc = new ClavierSession(embeddingDim: 32, hiddenDim: 16, numActions: 4, temperature: 3.5f);
        sessionSrc.TrainDecision("state_probe".AsSpan(), 1, learningRate: 0.2f);
        byte[] ckpt;
        using (var ms = new MemoryStream()) { sessionSrc.SaveCheckpoint(ms); ckpt = ms.ToArray(); }

        int dhOffset = FindSubsequence(ckpt, 0x48444C43u);
        Assert.True(dhOffset > 0);
        // DecisionHead layout: magic(4) + inDim(4) + hidDim(4) + actions(4) + temp(4) + w1[...]
        BinaryPrimitives.WriteSingleLittleEndian(ckpt.AsSpan(dhOffset + 20, 4), float.NaN);

        Assert.Throws<InvalidDataException>(() => session.LoadCheckpoint(new MemoryStream(ckpt)));
        Assert.Equal(1.0f, session.Temperature);
        Assert.Equal(origSessDec, session.Decide("state_probe".AsSpan()));
    }

    [Fact]
    public void DecideConformalSet_ExplicitQuantileCoincidentWithDefaultOrCalibrated_PreservedAcrossCheckpoint()
    {
        // 1. Uncalibrated session where explicit ConformalQuantile (0.95f) == 1 - ConformalAlpha (0.95f)
        using var uncal = new ClavierSession(embeddingDim: 32, hiddenDim: 16, numActions: 4);
        uncal.ConformalQuantile = 0.95f;

        byte[] uncalBytes;
        using (var ms = new MemoryStream()) { uncal.SaveCheckpoint(ms); uncalBytes = ms.ToArray(); }

        using var uncalRestored = new ClavierSession(embeddingDim: 32, hiddenDim: 16, numActions: 4);
        uncalRestored.LoadCheckpoint(new MemoryStream(uncalBytes));
        var setUncal = uncalRestored.DecideConformalSet("probe".AsSpan(), alpha: 0.25f);
        Assert.Equal(0.95f, setUncal.QuantileThreshold, precision: 5);

        // 2. Calibrated session where explicit ConformalQuantile equals calibrated quantile at ConformalAlpha
        float[] scores = [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 0.95f,
                          0.15f, 0.25f, 0.35f, 0.45f, 0.55f, 0.65f, 0.75f, 0.85f, 0.92f, 0.98f];
        using var calExplicit = new ClavierSession(embeddingDim: 32, hiddenDim: 16, numActions: 4);
        calExplicit.CalibrateConformal(scores, alpha: 0.05f);
        float calibratedQ = calExplicit.ConformalQuantile;
        calExplicit.ConformalQuantile = calibratedQ; // explicitly pinned to exact same float value

        byte[] calExplicitBytes;
        using (var ms = new MemoryStream()) { calExplicit.SaveCheckpoint(ms); calExplicitBytes = ms.ToArray(); }

        using var calExplicitRestored = new ClavierSession(embeddingDim: 32, hiddenDim: 16, numActions: 4);
        calExplicitRestored.LoadCheckpoint(new MemoryStream(calExplicitBytes));
        var setCalExplicit = calExplicitRestored.DecideConformalSet("probe".AsSpan(), alpha: 0.25f);
        Assert.Equal(calibratedQ, setCalExplicit.QuantileThreshold, precision: 5);

        // 3. Calibrated session WITHOUT explicit override still recalibrates dynamically for custom alpha
        using var calDynamic = new ClavierSession(embeddingDim: 32, hiddenDim: 16, numActions: 4);
        calDynamic.CalibrateConformal(scores, alpha: 0.05f);

        byte[] calDynamicBytes;
        using (var ms = new MemoryStream()) { calDynamic.SaveCheckpoint(ms); calDynamicBytes = ms.ToArray(); }

        using var calDynamicRestored = new ClavierSession(embeddingDim: 32, hiddenDim: 16, numActions: 4);
        calDynamicRestored.LoadCheckpoint(new MemoryStream(calDynamicBytes));
        var setCalDynamic = calDynamicRestored.DecideConformalSet("probe".AsSpan(), alpha: 0.25f);
        float expectedQ25 = ClavierCalibration.CalibrateQuantile(scores, 0.25f);
        Assert.Equal(expectedQ25, setCalDynamic.QuantileThreshold, precision: 5);

        // 4. Corrupt v3 explicit flag (> 1) is rejected atomically
        byte[] badFlag = (byte[])uncalBytes.Clone();
        // v3 layout: magic(4)+ver(4)+emb(4)+hid(4)+act(4)+q(4)+alpha(4) = offset 28
        badFlag[28] = 2;
        Assert.Throws<InvalidDataException>(() => calDynamicRestored.LoadCheckpoint(new MemoryStream(badFlag)));
        Assert.Equal(calibratedQ, calDynamicRestored.ConformalQuantile, precision: 5);
        Assert.Equal(expectedQ25, calDynamicRestored.DecideConformalSet("probe".AsSpan(), alpha: 0.25f).QuantileThreshold, precision: 5);

        // 5. Corrupt NaN inside calibration scores array is rejected atomically
        byte[] badCalScore = (byte[])calDynamicBytes.Clone();
        // v3 layout: 28 + explicit(1) + calCount(4) = offset 33 -> first float of calibration scores
        BinaryPrimitives.WriteSingleLittleEndian(badCalScore.AsSpan(33, 4), float.NaN);
        Assert.Throws<InvalidDataException>(() => uncalRestored.LoadCheckpoint(new MemoryStream(badCalScore)));
        Assert.Equal(0.95f, uncalRestored.ConformalQuantile, precision: 5);
        Assert.Equal(0.95f, uncalRestored.DecideConformalSet("probe".AsSpan(), alpha: 0.25f).QuantileThreshold, precision: 5);
    }

    [Fact]
    public void LoadCheckpoint_LegacyV2Format_RemainsBackwardCompatible()
    {
        using var ms = new MemoryStream();
        using (var writer = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            writer.Write(0x52564C43u); // "CLVR"
            writer.Write(2);           // legacy Version 2
            writer.Write(32);
            writer.Write(16);
            writer.Write(4);
            writer.Write(0.88f);       // ConformalQuantile
            writer.Write(0.05f);       // ConformalAlpha
            writer.Write(0);           // calCount = 0

            using var dh = new ClavierDecisionHead(32, 16, 4, temperature: 1.75f);
            var nh = new ClavierNoulHead(32, 16, temperature: 1.75f);
            var sh = new ClavierScoreHead(32, 16, temperature: 1.75f);
            dh.WriteWeights(writer);
            nh.WriteWeights(writer);
            sh.WriteWeights(writer);
        }
        ms.Position = 0;

        using var session = new ClavierSession(embeddingDim: 32, hiddenDim: 16, numActions: 4);
        session.LoadCheckpoint(ms);

        Assert.Equal(1.75f, session.Temperature);
        Assert.Equal(0.88f, session.ConformalQuantile, precision: 5);
        Assert.Equal(0.05f, session.ConformalAlpha, precision: 5);
        var set = session.DecideConformalSet("legacy v2".AsSpan(), alpha: 0.25f);
        Assert.Equal(0.88f, set.QuantileThreshold, precision: 5);
    }
}
