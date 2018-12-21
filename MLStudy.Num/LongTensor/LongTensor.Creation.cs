using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Num
{
    public static partial class Tensor
    {
        public static Tensor<long> Create(TensorData<long> data)
        {
            return new LongTensor(data);
        }

        public static Tensor<long> Create(long[] values)
        {
            var data = new TensorData<long>(values);
            return Create(data);
        }

        public static Tensor<long> Fill(long fillValue, params int[] shape)
        {
            var data = new TensorData<long>(fillValue, shape);
            return Create(data);
        }
    }
}
