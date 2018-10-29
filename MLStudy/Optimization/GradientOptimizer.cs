using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class GradientOptimizer
    {
        public double LearningRate { get; set; } = 0.01;

        public virtual Matrix GradientDescent(Matrix weights, Matrix gradient)
        {
            return weights - gradient * LearningRate;
        }

        public virtual Vector GradientDescent(Vector weights, Vector gradient)
        {
            return weights - gradient * LearningRate;
        }

        public virtual double GradientDescent(double bias, double gradient)
        {
            return bias - gradient * LearningRate;
        }
    }
}
