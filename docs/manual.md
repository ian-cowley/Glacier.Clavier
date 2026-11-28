# Glacier.Clavier User Manual

This manual documents `ClavierSession`, the three decision heads, split conformal gating, calibration utilities, and binary serialization in `Glacier.Clavier`.

---

## 1. Creating a Session & Custom State Encoders (`ClavierSession`)

```csharp
using Glacier.Clavier.Engine;

using var session = new ClavierSession(
    embeddingDim: 768,
    hiddenDim: 256,
    numActions: 64,
    temperature: 1.0f,
    encoder: null); // Optional ClavierStateEncoder delegate
```

You can plug any external embedding generator (such as `Glacier.Inference` or ONNX) into `ClavierStateEncoder(ReadOnlySpan<char> text, Span<float> destination)` so both state and candidate action strings encode using semantic representations instead of built-in feature hashing.

---

## 2. Three Decision Primitives

- **`Choice` (`Decide`, `DecideWithDistribution`, `DecideBatch`, `TrainDecision`):** Returns an 8-byte `ClavierDecision` (`ActionId`, `Confidence`). When `candidateActions` strings are passed, `(state, candidateAction[i])` pairs are jointly encoded so logits reflect action meaning independent of candidate order.
- **`Noul` (`Verify`, `VerifyWithDetails`, `VerifyBatch`, `TrainVerify`):** Returns an 8-byte `ClavierNoul` (`Probability`, `Affirmative` vs `threshold`).
- **`Score` (`Score`, `ScoreWithDetails`, `ScoreBatch`, `TrainScore`):** Returns an 8-byte `ClavierScore` (`Value` clamped to `[min, max]`, `Confidence`).

Adjust temperature scaling dynamically via `session.Temperature` (must be finite and `> 0`).

---

## 3. Split Conformal Gating (`CalibrateConformal` & `DecideConformalSet`)

Calibrate a holdout non-conformity quantile threshold $\hat{q}$ to decide automatically when System-1 can act alone versus escalating to a slower System-2 LLM:

```csharp
using Glacier.Clavier.Model;

float[] scores = new float[calibrationSamples];
ClavierCalibration.ComputeNonConformityScores(flatProbs, labels, numClasses, scores);

session.CalibrateConformal(scores, alpha: 0.05f); // >= 95% marginal coverage under exchangeability

ClavierPredictionSet predSet = session.DecideConformalSet(state.AsSpan(), alpha: 0.05f, actions);
if (predSet.IsSystem1Eligible) // Count == 1
{
    Console.WriteLine($"Fast action: {predSet.BestActionName} ({predSet.BestConfidence:P1})");
}
else
{
    Console.WriteLine($"Escalate ({predSet.EscalationReason}): {predSet.Count} candidates in C(x)");
}
```

Use `predSet.ToSummary()` for a zero-heap 24-byte unmanaged `ClavierPredictionSetSummary` containing `Count`, `BestActionId`, `BestConfidence`, `ActionMask` (64-bit bitmask), `QuantileThreshold`, and `EscalationReason` (`None`, `EmptyPredictionSet`, `AmbiguousPredictionSet`).

---

## 4. Calibration Metrics (`ClavierCalibration`)

- `ClavierCalibration.BrierScore(probabilities, targets)` — Proper quadratic scoring rule
- `ClavierCalibration.CategoricalCrossEntropy(logits, targets, labelSmoothing = 0.0f)` — Stable log-sum-exp loss with label smoothing
- `ClavierCalibration.ExpectedCalibrationError(confidences, predictions, groundTruth, numBins = 10)` — Binned ECE
- `ClavierCalibration.CalibrateQuantile(scores, alpha)` — Finite-sample $\lceil (n+1)(1-\alpha) \rceil / n$ quantile (`1.0f` fallback if $k > n$)

---

## 5. Binary Checkpoints (`.clvr`)

Persist and reload all three head weight matrices, temperatures, and calibration settings to any seekable/readable `Stream`:

```csharp
using (var fs = File.Create("policy.clvr"))
{
    session.SaveCheckpoint(fs);
}

using (var fs = File.OpenRead("policy.clvr"))
{
    session.LoadCheckpoint(fs);
}
```
