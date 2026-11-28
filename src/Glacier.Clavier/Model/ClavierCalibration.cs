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

        Span<float> diff = stackalloc float[Math.Min(probabilities.Length, 256)];
        double sumSq = 0.0;
        int offset = 0;
        while (offset < probabilities.Length)
        {
            int chunk = Math.Min(probabilities.Length - offset, 256);
            var pChunk = probabilities.Slice(offset, chunk);
            var tChunk = targets.Slice(offset, chunk);
            for (int i = 0; i < chunk; i++)
            {
                if (!float.IsFinite(pChunk[i]) || !float.IsFinite(tChunk[i]))
                    throw new ArgumentException("Probabilities and targets must contain only finite values.");
            }
            var dChunk = diff.Slice(0, chunk);
            System.Numerics.Tensors.TensorPrimitives.Subtract(pChunk, tChunk, dChunk);
            sumSq += SimdKernels.SumOfSquares(dChunk);
            offset += chunk;
        }
        return (float)(sumSq / probabilities.Length);
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
        if (!float.IsFinite(labelSmoothing) || labelSmoothing < 0f || labelSmoothing > 1f)
            throw new ArgumentOutOfRangeException(nameof(labelSmoothing), "Label smoothing must be finite and in [0, 1].");

        int maxIdx = 0;
        float maxLogit = logits[0];
        if (!float.IsFinite(maxLogit))
            throw new ArgumentException("Logits must contain only finite values.", nameof(logits));
        if (!float.IsFinite(targets[0]))
            throw new ArgumentException("Targets must contain only finite values.", nameof(targets));

        for (int c = 1; c < classes; c++)
        {
            float v = logits[c];
            if (!float.IsFinite(v))
                throw new ArgumentException("Logits must contain only finite values.", nameof(logits));
            if (!float.IsFinite(targets[c]))
                throw new ArgumentException("Targets must contain only finite values.", nameof(targets));
            if (v > maxLogit)
            {
                maxLogit = v;
                maxIdx = c;
            }
        }

        double sumOtherExp = 0.0;
        for (int c = 0; c < classes; c++)
        {
            if (c != maxIdx)
            {
                sumOtherExp += Math.Exp((double)logits[c] - maxLogit);
            }
        }

        double log1pOther = sumOtherExp == 0.0
            ? 0.0
            : (1.0 + sumOtherExp == 1.0 ? sumOtherExp : Math.Log(1.0 + sumOtherExp) * (sumOtherExp / ((1.0 + sumOtherExp) - 1.0)));
        double logSumExp = (double)maxLogit + log1pOther;

        float smoothVal = labelSmoothing / classes;
        double totalLoss = 0.0;

        for (int c = 0; c < classes; c++)
        {
            float targetVal = targets[c] * (1.0f - labelSmoothing) + smoothVal;
            totalLoss -= targetVal * ((double)logits[c] - logSumExp);
        }

        return (float)totalLoss;
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
        if (numBins <= 0)
            throw new ArgumentOutOfRangeException(nameof(numBins), "Number of bins must be positive.");

        int[]? rentedCounts = null;
        float[]? rentedAcc = null;
        float[]? rentedConf = null;

        Span<int> binCounts = numBins <= 256 ? stackalloc int[numBins] : (rentedCounts = ArrayPool<int>.Shared.Rent(numBins)).AsSpan(0, numBins);
        Span<float> binAccSums = numBins <= 256 ? stackalloc float[numBins] : (rentedAcc = ArrayPool<float>.Shared.Rent(numBins)).AsSpan(0, numBins);
        Span<float> binConfSums = numBins <= 256 ? stackalloc float[numBins] : (rentedConf = ArrayPool<float>.Shared.Rent(numBins)).AsSpan(0, numBins);
        binCounts.Clear();
        binAccSums.Clear();
        binConfSums.Clear();

        try
        {
            for (int i = 0; i < n; i++)
            {
                float rawConf = confidences[i];
                if (!float.IsFinite(rawConf))
                    throw new ArgumentException($"Confidence at index {i} must be finite.", nameof(confidences));

                float conf = Math.Clamp(rawConf, 0.0f, 1.0f);
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
        finally
        {
            if (rentedCounts != null) ArrayPool<int>.Shared.Return(rentedCounts);
            if (rentedAcc != null) ArrayPool<float>.Shared.Return(rentedAcc);
            if (rentedConf != null) ArrayPool<float>.Shared.Return(rentedConf);
        }
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
        if (alpha <= 0.0f || alpha >= 1.0f || !float.IsFinite(alpha))
            throw new ArgumentOutOfRangeException(nameof(alpha), alpha, "Significance level alpha must be in (0, 1).");

        int n = scores.Length;
        for (int i = 0; i < n; i++)
        {
            if (!float.IsFinite(scores[i]))
                throw new ArgumentException($"Calibration score at index {i} must be finite.", nameof(scores));
        }

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
        if (numClasses <= 0)
            throw new ArgumentOutOfRangeException(nameof(numClasses), "Number of classes must be positive.");
        int n = groundTruthLabels.Length;
        if (flatProbabilities.Length != checked(n * numClasses))
            throw new ArgumentException("flatProbabilities length must equal groundTruthLabels.Length * numClasses.");
        if (destinationScores.Length < n)
            throw new ArgumentException("destinationScores buffer is smaller than sample count.");

        for (int i = 0; i < n; i++)
        {
            int label = groundTruthLabels[i];
            if ((uint)label >= (uint)numClasses)
                throw new ArgumentOutOfRangeException(nameof(groundTruthLabels), $"Label {label} at index {i} out of bounds for {numClasses} classes.");

            float prob = flatProbabilities[i * numClasses + label];
            if (!float.IsFinite(prob))
                throw new ArgumentException($"Probability at sample {i} must be finite.", nameof(flatProbabilities));
            destinationScores[i] = Math.Clamp(1.0f - prob, 0.0f, 1.0f);
        }
    }
}
