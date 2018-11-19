using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Abstraction
{
    public interface ILayer
    {
        string Name { get; set; }

        Tensor PrepareTrain(Tensor input);
        Tensor Forward(Tensor input);
        Tensor Backward(Tensor error);
    }
}
