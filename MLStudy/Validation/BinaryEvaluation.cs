using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Validation
{
    public class BinaryEvaluation
    {
        public Tensor Y { get; private set; }
        public Tensor YHat { get; private set; }

        public double GetMeanSquareError()
        {
            return 0;
        }

        public double GetAccuracy()
        {
            return 0;
        }
    }
}
