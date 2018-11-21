using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Regularizations
{
    public class Lasso : WeightDecay, IRegularizer
    {
        public string Name => "Lasso";

        public Lasso(double strength)
        {
            Strength = strength;
        }

        public override double Decay(double weight)
        {
            return Strength;
        }

        public override Vector Decay(Vector weights)
        {
            return new Vector(weights.Length, Strength);
        }

        public override Matrix Decay(Matrix weights)
        {
            return new Matrix(weights.Rows, weights.Columns, Strength);
        }

        public void Regularize(Tensor parameters, Tensor gradient)
        {
            gradient.Add(Strength);
        }
    }
}
