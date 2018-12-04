using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Abstraction
{
    public interface IModel
    {
        TensorOld Predict(TensorOld input);
        void Step(TensorOld X, TensorOld y);
        double GetTrainLoss();
        double GetLoss(TensorOld y, TensorOld yHat);
        double GetTrainAccuracy();
        double GetAccuracy(TensorOld y, TensorOld yHat);
    }
}
