using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Regularizations
{
    public class Ridge: IRegularizer
    {
        public string Name => "Ridge";

        public double Strength { get; set; }

        public Ridge(double strength)
        {
            Strength = strength;
        }

        public void Regularize(TensorOld parameters, TensorOld gradient)
        {
            TensorOld.Apply(gradient, parameters, gradient, (a, b) => a + Strength * b);
        }
    }
}
