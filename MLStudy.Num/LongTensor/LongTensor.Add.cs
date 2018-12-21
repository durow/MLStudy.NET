using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Num
{
    //Add
    public partial class LongTensor
    {
        public override void AddLocal(long a)
        {
            TensorOperations.Instance.ApplyLocal(this, p => p + a);
        }

        public override Tensor<long> Add(long a)
        {
            var result = CreateSameShape();
            TensorOperations.Instance.Apply(this, ref result, p => p + a);
            return result;
        }

        public override void AddLocal(Tensor<long> a)
        {
            TensorOperations.Instance.ApplyLocal(this, a, (m, n) => m + n);
        }

        public override Tensor<long> Add(Tensor<long> a)
        {
            var result = CreateSameShape();
            TensorOperations.Instance.Apply(this, a, ref result, (m, n) => m + n);
            return result;
        }
    }
}
