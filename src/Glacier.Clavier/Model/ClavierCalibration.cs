namespace Glacier.Clavier.Model;

using System;
using System.Buffers;

/// <summary>
/// Mathematical calibration metrics and loss functions for System-1 decision heads.
/// Evaluates and optimizes probability calibration via strictly proper scoring rules (Brier score),
/// label smoothing cross-entropy, and Expected Calibration Error (ECE).
/// </summary>
public static class ClavierCalibration
{
    /// <summary>
    /// Computes the Brier score quadratic proper scoring rule: (1/N) * sum((p_k - y_k)^2).
    /// Returns 0.0 for perfect calibrated confidence.
    /// </summary>
    public static float BrierScore(ReadOnlySpan<float> probabilities, ReadOnlySpan<float> targets)
    {
        if (probabilities.Length != targets.Length)
            throw new ArgumentException($"Length mismatch: probabilities ({probabilities.Length}) != targets ({targets.Length}).");
        if (probabilities.IsEmpty) return 0f;

        Span<float> diff = stackalloc float[probabilities.Length <= 256 ? probabilities.Length : 0];
        if (diff.Length > 0)
        {
            System.Numerics.Tensors.TensorPrimitives.Subtract(probabilities, targets, diff);
            return SimdKernels.SumOfSquares(diff) / probabilities.Length;
        }

        float sumSq = 0f;
        for (int i = 0; i < probabilities.Length; i++)
        {
            float d = probabilities[i] - targets[i];
            sumSq += d * d;
        }
        return sumSq / probabilities.Length;
    }

    /// <summary>
    /// Computes multi-class categorical cross-entropy loss with uniform label smoothing.
    /// Smooths one-hot labels: y'_k = (1 - alpha) * y_k + alpha / K.
    /// </summary>
    public static float CategoricalCrossEntropy(
        ReadOnlySpan<float> logits,
        ReadOnlySpan<float> targets,
        float labelSmoothing = 0.0f)
    {
        int classes = logits.Length;
        if (classes == 0 || targets.Length != classes)
            throw new ArgumentException("Invalid logits/targets dimension.");

        float maxLogit = logits[0];
        for (int c = 1; c < classes; c++)
        {
            if (logits[c] > maxLogit) maxLogit = logits[c];
        }

        float sumExp = 0f;
        for (int c = 0; c < classes; c++)
        {
            sumExp += MathF.Exp(logits[c] - maxLogit);
        }

        float smoothVal = labelSmoothing / classes;
        float totalLoss = 0f;

        for (int c = 0; c < classes; c++)
        {
            float p = MathF.Exp(logits[c] - maxLogit) / sumExp;
            float targetVal = targets[c] * (1.0f - labelSmoothing) + smoothVal;
            totalLoss -= targetVal * MathF.Log(MathF.Max(p, 1e-9f));
        }

        return totalLoss;
    }

    /// <summary>
    /// Computes Expected Calibration Error (ECE) across partitioned confidence bins:
    /// sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|.
    /// </summary>
    public static float ExpectedCalibrationError(
        ReadOnlySpan<float> confidences,
        ReadOnlySpan<int> predictions,
        ReadOnlySpan<int> groundTruth,
        int numBins = 10)
    {
        int n = confidences.Length;
        if (n == 0 || predictions.Length != n || groundTruth.Length != n)
            throw new ArgumentException("Sample counts must be non-zero and matching.");

        int[] binCounts = new int[numBins];
        float[] binAccSums = new float[numBins];
        float[] binConfSums = new float[numBins];

        for (int i = 0; i < n; i++)
        {
            float conf = Math.Clamp(confidences[i], 0.0f, 1.0f);
            int binIdx = Math.Min((int)(conf * numBins), numBins - 1);

            binCounts[binIdx]++;
            binConfSums[binIdx] += conf;
            if (predictions[i] == groundTruth[i])
            {
                binAccSums[binIdx] += 1.0f;
            }
        }

        float ece = 0f;
        for (int b = 0; b < numBins; b++)
        {
            if (binCounts[b] > 0)
            {
                float avgAcc = binAccSums[b] / binCounts[b];
                float avgConf = binConfSums[b] / binCounts[b];
                float weight = (float)binCounts[b] / n;
                ece += weight * MathF.Abs(avgAcc - avgConf);
            }
        }

        return ece;
    }

    /// <summary>
    /// Computes the empirical conformal non-conformity quantile qHat given calibration non-conformity scores
    /// s_i in [0, 1] and significance level alpha in (0, 1):
    /// qHat = Quantile( ceil((n + 1)(1 - alpha)) / n ).
    /// Guarantees finite-sample coverage P(Y in C(X)) >= 1 - alpha under exchangeability.
    /// </summary>
    public static float CalibrateQuantile(ReadOnlySpan<float> scores, float alpha)
    {
        if (scores.IsEmpty)
            throw new ArgumentException("Calibration scores cannot be empty.", nameof(scores));
        if (alpha <= 0.0f || alpha >= 1.0f || float.IsNaN(alpha))
            throw new ArgumentOutOfRangeException(nameof(alpha), alpha, "Significance level alpha must be in (0, 1).");

        int n = scores.Length;
        int k = (int)MathF.Ceiling((n + 1) * (1.0f - alpha));

        // Finite-sample edge condition: if k > n, conservative coverage requires qHat = 1.0
        // so that all candidate classes are admitted (C(x) = Y), forcing System-2 escalation.
        if (k > n)
        {
            return 1.0f;
        }

        float[]? rented = null;
        Span<float> sorted = n <= 512
            ? stackalloc float[n]
            : (rented = ArrayPool<float>.Shared.Rent(n)).AsSpan(0, n);

        try
        {
            scores.CopyTo(sorted);
            sorted.Sort();

            int index = Math.Clamp(k - 1, 0, n - 1);
            float qHat = sorted[index];
            return Math.Clamp(qHat, 0.0f, 1.0f);
        }
        finally
        {
            if (rented != null)
            {
                ArrayPool<float>.Shared.Return(rented);
            }
        }
    }

    /// <summary>
    /// Computes canonical classification non-conformity scores s_i = 1 - P(y_i | x_i) across N samples.
    /// </summary>
    public static void ComputeNonConformityScores(
        ReadOnlySpan<float> flatProbabilities,
        ReadOnlySpan<int> groundTruthLabels,
        int numClasses,
        Span<float> destinationScores)
    {
        int n = groundTruthLabels.Length;
        if (flatProbabilities.Length != n * numClasses)
            throw new ArgumentException("flatProbabilities length must equal groundTruthLabels.Length * numClasses.");
        if (destinationScores.Length < n)
            throw new ArgumentException("destinationScores buffer is smaller than sample count.");

        for (int i = 0; i < n; i++)
        {
            int label = groundTruthLabels[i];
            if ((uint)label >= (uint)numClasses)
                throw new ArgumentOutOfRangeException(nameof(groundTruthLabels), $"Label {label} at index {i} out of bounds for {numClasses} classes.");

            float prob = flatProbabilities[i * numClasses + label];
            destinationScores[i] = Math.Clamp(1.0f - prob, 0.0f, 1.0f);
        }
    }
}
