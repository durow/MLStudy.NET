using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    class DerivativeFunctions
    {
        public static Matrix SquareError(Matrix yHat, Matrix y, bool raw = false)
        {
            if (raw)
                return 2 * (yHat - y);
            else
                return yHat - y;
        }

        public static Vector SquareError(Vector yHat, Vector y, bool raw = false)
        {
            if (raw)
                return 2 * (yHat - y);
            else
                return yHat - y;
        }

        public static Matrix LinearWeights(Matrix X)
        {
            return X.Transpose();
        }

        public static Matrix LinearWeights(Vector v)
        {
            return v.ToMatrix(true);
        }

        public static double LinearBias()
        {
            return 1;
        }

        public static Matrix ReLU(Matrix input)
        {
            return input.ApplyFunction(a => ReLU(a));
        }

        public static double ReLU(double d)
        {
            return d > 0 ? 1 : 0;
        }

        public static double Tanh(double d)
        {
            var g = Functions.Tanh(d);
            return 1 - Math.Pow(g, 2);
        }

        public static double TanhByResult(double tanhResult)
        {
            return 1 - Math.Pow(tanhResult, 2);
        }

        public static double Sigmoid(double input)
        {
            var o = Functions.Sigmoid(input);
            return o * (1 - o);
        }

        public static double SigmoidByResult(double sigmoidResult)
        {
            return sigmoidResult * (1 - sigmoidResult);
        }
    }
}
