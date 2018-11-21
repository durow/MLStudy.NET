using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Regularizations
{
    public class Ridge: WeightDecay, IRegularizer
    {
        public string Name => "Ridge";

        public Ridge(double strength)
        {
            Strength = strength;
        }
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

        public void Regularize(Tensor parameters, Tensor gradient)
        {
            Tensor.Apply(gradient, parameters, gradient, (a, b) => a + Strength * b);
        }
    }
}
