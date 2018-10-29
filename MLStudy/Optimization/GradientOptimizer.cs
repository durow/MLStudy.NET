using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class GradientOptimizer
    {
        public RegType RegType { get; set; } = RegType.None;
        public OptType OptType { get; set; } = OptType.None;
        public double LearningRate { get; set; } = 0.01;

        public Matrix MatrixHistory { get; private set; }
        public Vector VectorHistory { get; set; }
        public double ScalarHistory { get; set; }

        public GradientOptimizer()
        {
        }

        public Matrix GradientDescent(Matrix weights, Matrix gradient)
        {
            return weights - gradient * LearningRate;
        }

        public Vector GradientDescent(Vector weights, Vector gradient)
        {
            return weights - gradient * LearningRate;
        }

        public double GradientDescent(double bias, double gradient)
        {
            return bias - gradient * LearningRate;
        }
    }

    public enum RegType
    {
        None,
        L1,
        L2
    }

    public enum OptType
    {
        None,
        Momentum,
        Adam
    }
}
