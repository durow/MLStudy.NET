using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Optimization
{
    public class NormalDescent: GradientOptimizer
    {
        public override string Name => "NormalDescent";

        public override Matrix GradientDescent(Matrix weights, Matrix gradient)
        {
            return weights - gradient * LearningRate;
        }

        public override Vector GradientDescent(Vector weights, Vector gradient)
        {
            return weights - gradient * LearningRate;
        }

        public override double GradientDescent(double bias, double gradient)
        {
            return bias - gradient * LearningRate;
        }
    }
}
