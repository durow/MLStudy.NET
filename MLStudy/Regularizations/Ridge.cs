using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Regularizations
{
    public class Ridge: WeightDecay
    {
        public string Name => "Ridge";

        public override double GetValue(double weight)
        {
            return weight * Weight;
        }

        public override Vector GetValue(Vector weights)
        {
            return weights * Weight;
        }

        public override Matrix GetValue(Matrix weights)
        {
            return weights * Weight;
        }
    }
}
