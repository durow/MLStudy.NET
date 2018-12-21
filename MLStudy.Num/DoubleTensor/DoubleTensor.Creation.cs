using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Num
{
    public static partial class Tensor
    {
        public static Tensor<double> Create(TensorData<double> data)
        {
            return new DoubleTensor(data);
        }

        public static Tensor<double> Create(double[] values)
        {
            var data = new TensorData<double>(values);
            return Create(data);
        }

        public static Tensor<double> Create(double[] values, params int[] shape)
        {
            var data = new TensorData<double>(values, shape);
            return Create(data);
        }

        public static Tensor<double> Values(double[,] array)
        {
            return Values<double>(array);
        }
        public static Tensor<double> Values(double[,,] array)
        {
            return Values<double>(array);
        }

        public static Tensor<double> Values(double[,,,] array)
        {
            return Values<double>(array);
        }

        public static Tensor<double> Values(double[,,,,] array)
        {
            return Values<double>(array);
        }

        public static Tensor<double> Fill(double fillValue, params int[] shape)
        {
            var data = new TensorData<double>(fillValue, shape);
            return Create(data);
        }
    }
}
