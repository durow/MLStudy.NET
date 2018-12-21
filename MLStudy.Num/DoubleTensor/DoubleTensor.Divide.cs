using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Num
{
    public partial class DoubleTensor
    {
        public override void DivideLocal(double a)
        {
            TensorOperations.Instance.ApplyLocal(this, p => p / a);
        }

        public override Tensor<double> Divide(double a)
        {
            var result = CreateSameShape();
            TensorOperations.Instance.Apply(this, ref result, p => p / a);
            return result;
        }

        public override void DivideByLocal(double a)
        {
            TensorOperations.Instance.ApplyLocal(this, p => a / p);
        }

        public override Tensor<double> DivideBy(double a)
        {
            var result = CreateSameShape();
            TensorOperations.Instance.Apply(this, ref result, p => a / p);
            return result;
        }

        public override void DivideElementWiseLocal(Tensor<double> a)
        {
            TensorOperations.Instance.ApplyLocal(this, a, (m, n) => m / n);
        }

        public override Tensor<double> DivideElementWise(Tensor<double> a)
        {
            var result = a.CreateSameShape();
            TensorOperations.Instance.Apply(this, a, ref result, (m, n) => m / n);
            return result;
        }
    }
}
