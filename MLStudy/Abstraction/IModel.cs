using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Abstraction
{
    public interface IModel
    {
        Tensor Predict(Tensor input);
        void Step(Tensor X, Tensor y);
        double GetTrainLoss();
        double GetLoss(Tensor y, Tensor yHat);
        double GetTrainAccuracy();
        double GetAccuracy(Tensor y, Tensor yHat);
    }
}
