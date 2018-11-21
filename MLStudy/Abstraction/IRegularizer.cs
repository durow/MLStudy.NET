using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Abstraction
{
    public interface IRegularizer
    {
        void Regularize(Tensor parameters, Tensor gradient);
    }
}
