using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Abstraction
{
    public interface IEngine
    {
        Tensor Predict(Tensor input);
        void Step(Tensor X, Tensor y);
    }
}
