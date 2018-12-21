using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Num
{
    public partial class LongTensor
    {
        public override void MultipleLocal(long a)
        {
            TensorOperations.Instance.ApplyLocal(this, p => a * p);
        }

        public override Tensor<long> Multiple(long a)
        {
            var result = CreateSameShape();
            TensorOperations.Instance.Apply(this, ref result, p => a * p);
            return result;
        }

        public override Tensor<long> Multiple(Tensor<long> a)
        {
            var result = Tensor.Empty<long>(Shape[0], a.Shape[1]);
            TensorOperations.Instance.Multiple(this, a, ref result);
            return result;
        }

        public override void MultipleElementWiseLocal(Tensor<long> a)
        {
            TensorOperations.Instance.ApplyLocal(this, a, (m, n) => m * n);
        }

        public override Tensor<long> MultipleElementWise(Tensor<long> a)
        {
            var result = a.CreateSameShape();
            TensorOperations.Instance.Apply(this, a, ref result, (m, n) => m * n);
            return result;
        }
    }
}
