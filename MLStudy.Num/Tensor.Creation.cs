using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Num
{
    public static partial class Tensor
    {
        public static Tensor<T> Create<T>(TensorData<T> data)
            where T : struct
        {
            throw new NotImplementedException($"not implenment type {typeof(T).Name} yet!");
        }

        public static Tensor<T> Create<T>(T[] values)
            where T : struct
        {
            var data = new TensorData<T>(values);
            return Create(data);
        }

        public static Tensor<T> Create<T>(T[] values, params int[] shape)
            where T : struct
        {
            var data = new TensorData<T>(values, shape);
            return Create(data);
        }

        public static Tensor<T> Empty<T>(params int[] shape)
            where T : struct
        {
            var data = new TensorData<T>(shape);
            return Create(data);
        }

        public static Tensor<T> Fill<T>(T fillValue, params int[] shape)
            where T : struct
        {
            var data = new TensorData<T>(fillValue, shape);
            return Create(data);
        }
    }
}
