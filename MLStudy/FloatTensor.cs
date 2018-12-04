using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class FloatTensor : TensorGeneric<float>
    {
        public FloatTensor(params int[] shape)
            : base(shape) { }

        public override TensorGeneric<float> Add(float a)
        {
            return TensorGeneric.Add(this, a);
        }

        public override void AddLocal(float a)
        {
            TensorGeneric.Add(this, a, this);
        }

        public override TensorGeneric<float> GetSameShape()
        {
            return new FloatTensor(shape);
        }
    }
}
