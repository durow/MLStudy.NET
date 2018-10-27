using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class LossFunctions
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

        public static double CrossEntropy(double[] p, double[] pHat)
        {
            var result = 0d;

            for (int i = 0; i < p.Length; i++)
            {
                var temp = -p[i] * Math.Log(pHat[i]);
                result += temp;
            }

            return result;
        }
        public static double ErrorPercent(Vector yHat, Vector y)
        {
            yHat = (yHat - 0.5).ApplyFunction(Functions.IndicatorFunction);

            var sum = 0d;

            for (int i = 0; i < yHat.Length; i++)
            {
                if (yHat[i] != y[i])
                    sum++;
            }

            return sum / y.Length;
        }
    }
}
