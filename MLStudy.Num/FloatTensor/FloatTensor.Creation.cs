using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Num
{
    public static partial class Tensor
    {
        public static Tensor<float> Create(TensorData<float> data)
        {
            return new FloatTensor(data);
        }

        public static Tensor<float> Create(float[] values)
        {
            var data = new TensorData<float>(values);
            return Create(data);
        }

        public static Tensor<float> Create(float[] values, params int[] shape)
        {
            var data = new TensorData<float>(values, shape);
            return Create(data);
        }

        public static Tensor<float> Values(float[,] array)
        {
            return Values<float>(array);
        }
        public static Tensor<float> Values(float[,,] array)
        {
            return Values<float>(array);
        }

        public static Tensor<float> Values(float[,,,] array)
        {
            return Values<float>(array);
        }

        public static Tensor<float> Values(float[,,,,] array)
        {
            return Values<float>(array);
        }

        public static Tensor<float> Empty(params int[] shape)
        {
            var data = new TensorData<float>(shape);
            return Create(data);
        }

        public static Tensor<float> Ones(params int[] shape)
        {
            return Fill(1f, shape);
        }

        public static Tensor<float> Zeros(params int[] shape)
        {
            return Fill(0f, shape);
        }

        public static Tensor<float> Fill(float fillValue, params int[] shape)
        {
            var data = new TensorData<float>(fillValue, shape);
            return Create(data);
        }
    }
}
