using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public sealed partial class Tensor
    {
        public static Tensor WithValues(float value, params int[] shape)
        {
            var result = Create(shape);
            result.Values.SetAll(value);
            return result;
        }

        public static Tensor Zeros(params int[] shape)
        {
            return WithValues(0f, shape);
        }

        public static Tensor Ones(params int[] shape)
        {
            return WithValues(1f, shape);
        }

        public static Tensor<T> WithValues<T>(T value, params int[] shape)
            where T : struct
        {
            var result = Create<T>(shape);
            result.Values.SetAll(value);
            return result;
        }

        public static Tensor<T> Zeros<T>(params int[] shape)
            where T : struct
        {
            var type = typeof(T);

            if (type == typeof(float))
            {
                var result = WithValues<float>(0f, shape);
                return (Tensor<T>)(object)result;
            }

            if (type == typeof(double))
            {
                var result = WithValues(0d, shape);
                return (Tensor<T>)(object)result;
            }

            if (type == typeof(int))
            {
                var result = WithValues(0, shape);
                return (Tensor<T>)(object)result;
            }

            throw new NotImplementedException($"not implemented type {type.Name}");
        }

        public static Tensor<T> Ones<T>(params int[] shape)
            where T : struct
        {
            var type = typeof(T);

            if (type == typeof(float))
            {
                var result = WithValues(1f, shape);
                return (Tensor<T>)(object)result;
            }

            if (type == typeof(double))
            {
                var result = WithValues(1d, shape);
                return (Tensor<T>)(object)result;
            }

            if (type == typeof(int))
            {
                var result = WithValues(1, shape);
                return (Tensor<T>)(object)result;
            }

            throw new NotImplementedException($"not implemented type {type.Name}");
        }

        public static Tensor Create(params int[] shape)
        {
            return new Tensor(shape);
        }

        public static Tensor<T> Create<T>(params int[] shape)
            where T : struct
        {
            var type = typeof(T);

            if (type == typeof(float))
                return (Tensor<T>)(object)(new Tensor(shape));

            throw new NotImplementedException($"not implemented type {type.Name}");
        }

        public static void Add(Tensor<float> a, float b, Tensor<float> result)
        {
            Apply(a, result, p => p + b);
        }

        public static Tensor<float> Add(Tensor<float> a, float b)
        {
            var result = a.GetSameShape();
            Add(a, b, result);
            return result;
        }

        public static Tensor<T> Add<T>(Tensor<T> a, T b)
            where T : struct
        {
            var result = a.GetSameShape();
            Add(a, b, result);
            return result;
        }

        public static void Add<T>(Tensor<T> a, T b, Tensor<T> result)
            where T : struct
        {
            throw new NotImplementedException($"not implement Add of {typeof(T).Name}");
        }

        public static void Apply<T>(Tensor<T> a, Tensor<T> result, Func<T, T> function)
            where T : struct
        {
            var end = a.Values.startIndex + a.Values.Count;
            for (int i = a.Values.startIndex; i < end; i++)
            {
                result.Values.rawValues[i] = function(a.Values.rawValues[i]);
            }
        }

        public static void Apply<T>(Tensor<T> a, Tensor<T> b, Tensor<T> result, Func<T, T, T> function)
            where T : struct
        {
            var end = a.Values.startIndex + a.Values.Count;
            for (int i = a.Values.startIndex; i < end; i++)
            {
                result.Values.rawValues[i] = function(a.Values.rawValues[i], b.Values.rawValues[i]);
            }
        }
    }
}
