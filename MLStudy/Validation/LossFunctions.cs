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
            return error * error;
        }

        public static double CrossEntropy(Vector p, Vector pHat)
        {
            var result = 0d;

            for (int i = 0; i < p.Length; i++)
            {
                if (p[i] == 0)
                    continue;

                var temp = -p[i] * Math.Log(pHat[i]);
                result += temp;
            }

            return result;
        }

        public static double Logistic(Vector yHat, Vector y)
        {
            var sum = 0d;
            var crossEntropy = 0d;
            for (int i = 0; i < y.Length; i++)
            {
                if (y[i] == 1)
                    crossEntropy = -Math.Log(yHat[i]);
                else
                    crossEntropy = -Math.Log(1 - yHat[i]);

                sum += crossEntropy;
            }
            return sum / y.Length;
        }

        public static double ErrorPercent(Vector yHat, Vector y)
        {
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
