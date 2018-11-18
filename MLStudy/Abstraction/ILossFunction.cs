using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Abstraction
{
    public interface ILossFunction
    {
        double GetLoss(Tensor y, Tensor yHat);
        Tensor GetGradient(Tensor y, Tensor yHat);
    }
}
