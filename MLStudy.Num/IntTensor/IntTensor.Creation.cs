using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Num
{
    public static partial class Tensor
    {
        public static Tensor<int> Create(TensorData<int> data)
        {
            return new IntTensor(data);
        }

        public static Tensor<int> Create(int[] values)
        {
            var data = new TensorData<int>(values);
            return Create(data);
        }

        public static Tensor<int> Create(int[] values, params int[] shape)
        {
            var data = new TensorData<int>(values, shape);
            return Create(data);
        }

        public static Tensor<int> Values(int[,] array)
        {
            return Values<int>(array);
        }
        public static Tensor<int> Values(int[,,] array)
        {
            return Values<int>(array);
        }

        public static Tensor<int> Values(int[,,,] array)
        {
            return Values<int>(array);
        }

        public static Tensor<int> Values(int[,,,,] array)
        {
            return Values<int>(array);
        }

        public static Tensor<int> Fill(int fillValue, params int[] shape)
        {
            var data = new TensorData<int>(fillValue, shape);
            return Create(data);
        }
    }
}
