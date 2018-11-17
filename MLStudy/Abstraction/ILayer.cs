using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Abstraction
{
    public interface ILayer
    {
        Tensor Forward(Tensor input);
        Tensor Backward(Tensor error);
    }
}
