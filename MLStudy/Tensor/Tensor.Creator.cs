using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public sealed partial class Tensor
    {
        public static Tensor FromData(float[] data)
        {
            return new Tensor(data);
        }
        
        public static Tensor Empty(params int[] shape)
        {
            return new Tensor(shape);
        }

        public static Tensor InitValue(float value, params int[] shape)
        {
            var result = Empty(shape);
            result.Values.SetAll(value);
            return result;
        }

        public static Tensor Zeros(params int[] shape)
        {
            return InitValue(0f, shape);
        }

        public static Tensor Ones(params int[] shape)
        {
            return InitValue(1f, shape);
        }

        public static Tensor<T> FromData<T>(T[] data)
            where T : struct
        {
            throw new NotImplementedException();
        }

        public static Tensor<T> InitValue<T>(T value, params int[] shape)
            where T : struct
        {
            var result = Empty<T>(shape);
            result.Values.SetAll(value);
            return result;
        }

        public static Tensor<T> Zeros<T>(params int[] shape)
            where T : struct
        {
            var type = typeof(T);

            if (type == typeof(float))
            {
                var result = InitValue<float>(0f, shape);
                return (Tensor<T>)(object)result;
            }

            if (type == typeof(double))
            {
                var result = InitValue(0d, shape);
                return (Tensor<T>)(object)result;
            }

            if (type == typeof(int))
            {
                var result = InitValue(0, shape);
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
                var result = InitValue(1f, shape);
                return (Tensor<T>)(object)result;
            }

            if (type == typeof(double))
            {
                var result = InitValue(1d, shape);
                return (Tensor<T>)(object)result;
            }

            if (type == typeof(int))
            {
                var result = InitValue(1, shape);
                return (Tensor<T>)(object)result;
            }

            throw new NotImplementedException($"not implemented type {type.Name}");
        }

        public static Tensor<T> Empty<T>(params int[] shape)
            where T : struct
        {
            var type = typeof(T);

            if (type == typeof(float))
                return (Tensor<T>)(object)(new Tensor(shape));

            throw new NotImplementedException($"not implemented type {type.Name}");
        }
    }
}
