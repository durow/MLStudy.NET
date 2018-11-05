using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Abstraction
{
    public abstract class CNNLayer
    {
        public abstract Tensor3 Forward(Tensor3 input);
        public abstract Tensor3 Backward(Tensor3 outputError);
    }
}
