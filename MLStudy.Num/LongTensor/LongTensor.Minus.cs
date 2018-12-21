using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Num
{
    public partial class LongTensor
    {
        public override void MinusLocal(long a)
        {
            TensorOperations.Instance.ApplyLocal(this, p => p - a);
        }

        public override Tensor<long> Minus(long a)
        {
            var result = CreateSameShape();
            TensorOperations.Instance.Apply(this, ref result, p => p - a);
            return result;
        }

        public override void MunusByLocal(long a)
        {
            TensorOperations.Instance.ApplyLocal(this, p => a - p);
        }

        public override Tensor<long> MinusBy(long a)
        {
            var result = CreateSameShape();
            TensorOperations.Instance.Apply(this, ref result, p => a - p);
            return result;
        }

        public override void MinusLocal(Tensor<long> a)
        {
            TensorOperations.Instance.ApplyLocal(this, a, (m, n) => m - n);
        }

        public override Tensor<long> Minus(Tensor<long> a)
        {
            var result = CreateSameShape();
            TensorOperations.Instance.Apply(this, a, ref result, (m, n) => m - n);
            return result;
        }
    }
}
