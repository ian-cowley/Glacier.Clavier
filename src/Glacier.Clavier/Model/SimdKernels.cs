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
        if (Avx512F.IsSupported && x.Length >= Vector512<float>.Count && x.Length == y.Length)
        {
            int vCount = Vector512<float>.Count;
            int i = 0;
            int limit = x.Length - vCount;
            Vector512<float> sum512 = Vector512<float>.Zero;

            ref float xRef = ref Unsafe.AsRef(in x[0]);
            ref float yRef = ref Unsafe.AsRef(in y[0]);

            while (i <= limit)
            {
                var vx = Vector512.LoadUnsafe(ref Unsafe.Add(ref xRef, i));
                var vy = Vector512.LoadUnsafe(ref Unsafe.Add(ref yRef, i));
                sum512 = Avx512F.FusedMultiplyAdd(vx, vy, sum512);
                i += vCount;
            }

            float total = Vector512.Sum(sum512);
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
