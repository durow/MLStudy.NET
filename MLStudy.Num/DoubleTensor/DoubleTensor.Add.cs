using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Num
{
    //Add
    public partial class DoubleTensor
    {
        public override void AddLocal(double a)
        {
            TensorOperations.Instance.ApplyLocal(this, p => p + a);
        }

        public override Tensor<double> Add(double a)
        {
            var result = CreateSameShape();
            TensorOperations.Instance.Apply(this, ref result, p => p + a);
            return result;
        }

        public override void AddLocal(Tensor<double> a)
        {
            TensorOperations.Instance.ApplyLocal(this, a, (m, n) => m + n);
        }

        public override Tensor<double> Add(Tensor<double> a)
        {
            var result = CreateSameShape();
            TensorOperations.Instance.Apply(this, a, ref result, (m, n) => m + n);
            return result;
        }
    }
}
