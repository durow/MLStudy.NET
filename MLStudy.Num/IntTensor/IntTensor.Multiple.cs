using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Num
{
    public partial class IntTensor
    {
        public override void MultipleLocal(int a)
        {
            TensorOperations.Instance.ApplyLocal(this, p => a * p);
        }

        public override Tensor<int> Multiple(int a)
        {
            var result = CreateSameShape();
            TensorOperations.Instance.Apply(this, ref result, p => a * p);
            return result;
        }

        public override Tensor<int> Multiple(Tensor<int> a)
        {
            var result = Tensor.Empty<int>(Shape[0], a.Shape[1]);
            TensorOperations.Instance.Multiple(this, a, ref result);
            return result;
        }

        public override void MultipleElementWiseLocal(Tensor<int> a)
        {
            TensorOperations.Instance.ApplyLocal(this, a, (m, n) => m * n);
        }

        public override Tensor<int> MultipleElementWise(Tensor<int> a)
        {
            var result = a.CreateSameShape();
            TensorOperations.Instance.Apply(this, a, ref result, (m, n) => m * n);
            return result;
        }
    }
}
