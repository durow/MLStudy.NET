using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class Loss
    {
        public static double StandardError(Vector yHat, Vector y)
        {
            return Math.Sqrt(MeanSquareError(yHat, y));
        }

        public static double MeanSquareError(Vector yHat, Vector y)
        {
            return SquareError(yHat, y) / y.Length;
        }

        public static double SquareError(Vector yHat, Vector y)
        {
            var error = yHat - y;
            return Tensor.MultipleAsMatrix(error, error);
        }

        public static double CrossEntropy(Vector yHat, Vector y)
        {
            return 0;
        }
    }
}
