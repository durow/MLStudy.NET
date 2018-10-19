using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class Gradient
    {
        //weightGradient = X^T(yHat-y)/y.Length
        public static Vector LinearWeights(Matrix X, Vector yHat, Vector y)
        {
            var gradient = X.Transpose() * (yHat - y) / y.Length;
            return gradient.GetColumn(0);
        }

        //biasGradient = (yHat-y)/SampleNumber
        public static double LinearBias(Vector yHat, Vector y)
        {
            return (yHat - y).Mean();
        }

        public static double LeastSquareError(Matrix X, Vector yHat, Vector y)
        {
            return 0;
        }

        public static double CrossEntropy()
        {
            return 0;
        }

        public static double Softmax()
        {
            return 0;
        }

        public static double Sigmoid()
        {
            return 0;
        }

        public static double ReLU()
        {
            return 0;
        }

        public static double Tanh()
        {
            return 0;
        }
    }
}
