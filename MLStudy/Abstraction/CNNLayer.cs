using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Abstraction
{
    public abstract class CNNLayer
    {
        public abstract Tensor Forward(Tensor input);
        public abstract Tensor Backward(Tensor outputError);
    }
}
