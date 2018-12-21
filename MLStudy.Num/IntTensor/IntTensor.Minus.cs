using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Num
{
    public partial class IntTensor
    {
        public override void MinusLocal(int a)
        {
            TensorOperations.Instance.ApplyLocal(this, p => p - a);
        }

        public override Tensor<int> Minus(int a)
        {
            var result = CreateSameShape();
            TensorOperations.Instance.Apply(this, ref result, p => p - a);
            return result;
        }

        public override void MunusByLocal(int a)
        {
            TensorOperations.Instance.ApplyLocal(this, p => a - p);
        }

        public override Tensor<int> MinusBy(int a)
        {
            var result = CreateSameShape();
            TensorOperations.Instance.Apply(this, ref result, p => a - p);
            return result;
        }

        public override void MinusLocal(Tensor<int> a)
        {
            TensorOperations.Instance.ApplyLocal(this, a, (m, n) => m - n);
        }

        public override Tensor<int> Minus(Tensor<int> a)
        {
            var result = CreateSameShape();
            TensorOperations.Instance.Apply(this, a, ref result, (m, n) => m - n);
            return result;
        }
    }
}
