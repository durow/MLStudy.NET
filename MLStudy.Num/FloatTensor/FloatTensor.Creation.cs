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

        public static Tensor<float> Empty(params int[] shape)
        {
            var data = new TensorData<float>(shape);
            return Create(data);
        }

        public static Tensor<float> Fill(float fillValue, params int[] shape)
        {
            var data = new TensorData<float>(fillValue, shape);
            return Create(data);
        }
    }
}
