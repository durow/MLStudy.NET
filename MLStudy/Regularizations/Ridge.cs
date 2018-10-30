using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Regularizations
{
    public class Ridge: WeightDecay
    {
        public string Name => "Ridge";

        public override double Decay(double weight)
        {
            return weight * Strength;
        }

        public override Vector Decay(Vector weights)
        {
            return weights * Strength;
        }

        public override Matrix Decay(Matrix weights)
        {
            return weights * Strength;
        }
    }
}
