using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Regularizations
{
    public class Lasso : IRegularizer
    {
        public string Name => "Lasso";
        public double Strength { get; set; }

        public Lasso(double strength)
        {
            Strength = strength;
        }

        public void Regularize(Tensor parameters, Tensor gradient)
        {
            gradient.Add(Strength);
        }
    }
}
