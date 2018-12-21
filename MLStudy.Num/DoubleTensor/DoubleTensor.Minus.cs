using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Num
{
    public partial class DoubleTensor
    {
        public override void MinusLocal(double a)
        {
            TensorOperations.Instance.ApplyLocal(this, p => p - a);
        }

        public override Tensor<double> Minus(double a)
        {
            var result = CreateSameShape();
            TensorOperations.Instance.Apply(this, ref result, p => p - a);
            return result;
        }

        public override void MunusByLocal(double a)
        {
            TensorOperations.Instance.ApplyLocal(this, p => a - p);
        }

        public override Tensor<double> MinusBy(double a)
        {
            var result = CreateSameShape();
            TensorOperations.Instance.Apply(this, ref result, p => a - p);
            return result;
        }

        public override void MinusLocal(Tensor<double> a)
        {
            TensorOperations.Instance.ApplyLocal(this, a, (m, n) => m - n);
        }

        public override Tensor<double> Minus(Tensor<double> a)
        {
            var result = CreateSameShape();
            TensorOperations.Instance.Apply(this, a, ref result, (m, n) => m - n);
            return result;
        }
    }
}
