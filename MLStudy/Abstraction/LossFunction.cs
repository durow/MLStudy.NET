using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Abstraction
{
    public abstract class LossFunction
    {
        public Tensor ForwardOutput { get; protected set; }
        public Tensor BackwardOutput { get; protected set; }

        public double GetLoss()
        {
            return ForwardOutput.Mean();
        }

        public abstract void PrepareTrain(Tensor y, Tensor yHat);
        public abstract void Compute(Tensor y, Tensor yHat);
    }
}
