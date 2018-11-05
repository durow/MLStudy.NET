using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public abstract class GradientOptimizer
    {
        public abstract string Name { get; }
        public double LearningRate { get; set; } = 0.01;

        public abstract Tensor3 GradientDescent(Tensor3 weights, Tensor3 gradient);

        public abstract Matrix GradientDescent(Matrix weights, Matrix gradient);

        public abstract Vector GradientDescent(Vector weights, Vector gradient);

        public abstract double GradientDescent(double bias, double gradient);
    }
}
