using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class Gradient
    {
        public static (Vector, double) LinearSquareError(Matrix X, Vector y, Vector yHat)
        {
            //weightGradient = X^T(yHat-y)/y.Length
            var gradientWeights = X.Transpose() * (yHat - y) / y.Length;
            //biasGradient = (yHat-y)/SampleNumber
            var gradientBias = (yHat - y).Mean();
            return (gradientWeights.ToVector(), gradientBias);
        }

        public static (Vector, double) LinearSigmoidCrossEntropy(Matrix X, Vector y, Vector yHat)
        {
            //gradient of weights = X^T*(yHat-y)/m     m is sample number
            var gradientWeights = (X.Transpose() * (yHat - y)) / y.Length;
            //biasGradient = (yHat-y)/SampleNumber
            var gradientBias = (yHat - y).Mean();
            return (gradientWeights.ToVector(), gradientBias);
        }

        public static (Matrix, Vector) LinearSoftmaxCrossEntropy(Matrix X, Matrix y, Matrix yHat)
        {
            var linearError = yHat - y;
            var v = new Vector(linearError.Columns, 1);
            var gradientBias = (v * linearError).ToVector() / linearError.Rows;
            var gradientWeights = X.Transpose() * linearError / linearError.Rows;

            return (gradientWeights, gradientBias);
        }

        public static Vector LinearL1(Vector weights, double eta)
        {
            return (new Vector(weights.Length) + 1) * eta;
        }

        public static Vector LinearL2(Vector weights, double eta)
        {
            return weights * eta;
        }
    }
}
