namespace Glacier.Clavier.Model;

using System;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

/// <summary>
/// Hardware SIMD acceleration kernels using System.Numerics.Tensors and AVX-512 FMA intrinsics.
/// </summary>
public static class SimdKernels
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Dot(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
        if (x.Length == y.Length && Avx512F.IsSupported && x.Length >= Vector512<float>.Count)
        {
            int vCount = Vector512<float>.Count;
            int step4 = vCount * 4;
            int i = 0;

            ref float xRef = ref Unsafe.AsRef(in x[0]);
            ref float yRef = ref Unsafe.AsRef(in y[0]);
            Vector512<float> sum0 = Vector512<float>.Zero;

            if (x.Length >= step4)
            {
                Vector512<float> sum1 = Vector512<float>.Zero;
                Vector512<float> sum2 = Vector512<float>.Zero;
                Vector512<float> sum3 = Vector512<float>.Zero;
                int limit4 = x.Length - step4;
                while (i <= limit4)
                {
                    sum0 = Avx512F.FusedMultiplyAdd(Vector512.LoadUnsafe(ref Unsafe.Add(ref xRef, i)), Vector512.LoadUnsafe(ref Unsafe.Add(ref yRef, i)), sum0);
                    sum1 = Avx512F.FusedMultiplyAdd(Vector512.LoadUnsafe(ref Unsafe.Add(ref xRef, i + vCount)), Vector512.LoadUnsafe(ref Unsafe.Add(ref yRef, i + vCount)), sum1);
                    sum2 = Avx512F.FusedMultiplyAdd(Vector512.LoadUnsafe(ref Unsafe.Add(ref xRef, i + vCount * 2)), Vector512.LoadUnsafe(ref Unsafe.Add(ref yRef, i + vCount * 2)), sum2);
                    sum3 = Avx512F.FusedMultiplyAdd(Vector512.LoadUnsafe(ref Unsafe.Add(ref xRef, i + vCount * 3)), Vector512.LoadUnsafe(ref Unsafe.Add(ref yRef, i + vCount * 3)), sum3);
                    i += step4;
                }
                sum0 = (sum0 + sum1) + (sum2 + sum3);
            }

            int limit1 = x.Length - vCount;
            while (i <= limit1)
            {
                sum0 = Avx512F.FusedMultiplyAdd(Vector512.LoadUnsafe(ref Unsafe.Add(ref xRef, i)), Vector512.LoadUnsafe(ref Unsafe.Add(ref yRef, i)), sum0);
                i += vCount;
            }

            float total = Vector512.Sum(sum0);
            while (i < x.Length)
            {
                total += x[i] * y[i];
                i++;
            }
            return total;
        }

        if (x.Length == y.Length && Fma.IsSupported && Avx.IsSupported && x.Length >= Vector256<float>.Count)
        {
            int vCount = Vector256<float>.Count;
            int step4 = vCount * 4;
            int i = 0;

            ref float xRef = ref Unsafe.AsRef(in x[0]);
            ref float yRef = ref Unsafe.AsRef(in y[0]);
            Vector256<float> sum0 = Vector256<float>.Zero;

            if (x.Length >= step4)
            {
                Vector256<float> sum1 = Vector256<float>.Zero;
                Vector256<float> sum2 = Vector256<float>.Zero;
                Vector256<float> sum3 = Vector256<float>.Zero;
                int limit4 = x.Length - step4;
                while (i <= limit4)
                {
                    sum0 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref Unsafe.Add(ref xRef, i)), Vector256.LoadUnsafe(ref Unsafe.Add(ref yRef, i)), sum0);
                    sum1 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref Unsafe.Add(ref xRef, i + vCount)), Vector256.LoadUnsafe(ref Unsafe.Add(ref yRef, i + vCount)), sum1);
                    sum2 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref Unsafe.Add(ref xRef, i + vCount * 2)), Vector256.LoadUnsafe(ref Unsafe.Add(ref yRef, i + vCount * 2)), sum2);
                    sum3 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref Unsafe.Add(ref xRef, i + vCount * 3)), Vector256.LoadUnsafe(ref Unsafe.Add(ref yRef, i + vCount * 3)), sum3);
                    i += step4;
                }
                sum0 = (sum0 + sum1) + (sum2 + sum3);
            }

            int limit1 = x.Length - vCount;
            while (i <= limit1)
            {
                sum0 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref Unsafe.Add(ref xRef, i)), Vector256.LoadUnsafe(ref Unsafe.Add(ref yRef, i)), sum0);
                i += vCount;
            }

            float total = Vector256.Sum(sum0);
            while (i < x.Length)
            {
                total += x[i] * y[i];
                i++;
            }
            return total;
        }

        return TensorPrimitives.Dot(x, y);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float SumOfSquares(ReadOnlySpan<float> x)
    {
        if (Avx512F.IsSupported && x.Length >= Vector512<float>.Count)
        {
            int vCount = Vector512<float>.Count;
            int i = 0;
            int limit = x.Length - vCount;
            Vector512<float> sum512 = Vector512<float>.Zero;

            ref float xRef = ref Unsafe.AsRef(in x[0]);

            while (i <= limit)
            {
                var vx = Vector512.LoadUnsafe(ref Unsafe.Add(ref xRef, i));
                sum512 = Avx512F.FusedMultiplyAdd(vx, vx, sum512);
                i += vCount;
            }

            float total = Vector512.Sum(sum512);
            while (i < x.Length)
            {
                total += x[i] * x[i];
                i++;
            }
            return total;
        }

        return TensorPrimitives.SumOfSquares(x);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Multiply(Span<float> x, float scalar)
    {
        TensorPrimitives.Multiply(x, scalar, x);
    }
}
