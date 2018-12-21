using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Num
{
    //Add
    public partial class IntTensor
    {
        public override void AddLocal(int a)
        {
            TensorOperations.Instance.ApplyLocal(this, p => p + a);
        }

        public override Tensor<int> Add(int a)
        {
            var result = CreateSameShape();
            TensorOperations.Instance.Apply(this, ref result, p => p + a);
            return result;
        }

        public override void AddLocal(Tensor<int> a)
        {
            TensorOperations.Instance.ApplyLocal(this, a, (m, n) => m + n);
        }

        public override Tensor<int> Add(Tensor<int> a)
        {
            var result = CreateSameShape();
            TensorOperations.Instance.Apply(this, a, ref result, (m, n) => m + n);
            return result;
        }
    }
}
