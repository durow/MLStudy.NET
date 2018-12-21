using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Num
{
    public partial class IntTensor
    {
        public override void DivideLocal(int a)
        {
            TensorOperations.Instance.ApplyLocal(this, p => p / a);
        }

        public override Tensor<int> Divide(int a)
        {
            var result = CreateSameShape();
            TensorOperations.Instance.Apply(this, ref result, p => p / a);
            return result;
        }

        public override void DivideByLocal(int a)
        {
            TensorOperations.Instance.ApplyLocal(this, p => a / p);
        }

        public override Tensor<int> DivideBy(int a)
        {
            var result = CreateSameShape();
            TensorOperations.Instance.Apply(this, ref result, p => a / p);
            return result;
        }

        public override void DivideElementWiseLocal(Tensor<int> a)
        {
            TensorOperations.Instance.ApplyLocal(this, a, (m, n) => m / n);
        }

        public override Tensor<int> DivideElementWise(Tensor<int> a)
        {
            var result = a.CreateSameShape();
            TensorOperations.Instance.Apply(this, a, ref result, (m, n) => m / n);
            return result;
        }
    }
}
