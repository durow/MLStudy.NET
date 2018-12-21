using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Num
{
    public partial class DoubleTensor
    {
        public override void MultipleLocal(double a)
        {
            TensorOperations.Instance.ApplyLocal(this, p => a * p);
        }

        public override Tensor<double> Multiple(double a)
        {
            var result = CreateSameShape();
            TensorOperations.Instance.Apply(this, ref result, p => a * p);
            return result;
        }

        public override Tensor<double> Multiple(Tensor<double> a)
        {
            var result = Tensor.Empty<double>(Shape[0], a.Shape[1]);
            TensorOperations.Instance.Multiple(this, a, ref result);
            return result;
        }

        public override void MultipleElementWiseLocal(Tensor<double> a)
        {
            TensorOperations.Instance.ApplyLocal(this, a, (m, n) => m * n);
        }

        public override Tensor<double> MultipleElementWise(Tensor<double> a)
        {
            var result = a.CreateSameShape();
            TensorOperations.Instance.Apply(this, a, ref result, (m, n) => m * n);
            return result;
        }
    }
}
