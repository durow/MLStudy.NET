using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public sealed partial class Tensor : Tensor<float>
    {
        public Tensor(params int[] shape)
            : base(shape) { }

        public override Tensor<float> Add(float a)
        {
            return Tensor.Add(this, a);
        }

        public override void AddLocal(float a)
        {
            Tensor.Add(this, a, this);
        }

        public override Tensor<float> GetSameShape()
        {
            return new Tensor(shape);
        }
    }
}
