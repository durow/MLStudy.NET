using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class Gradient
    {
        #region Linear Models

        public static (Vector, double) LinearRegressionLoss(Matrix X, Vector y, Vector yHat)
        {
            //weightGradient = X^T(yHat-y)/y.Length
            var gradientWeights = X.Transpose() * (yHat - y) / y.Length;
            //biasGradient = (yHat-y)/SampleNumber
            var gradientBias = (yHat - y).Mean();
            return (gradientWeights.ToVector(), gradientBias);
        }

        public static (Vector, double) LogisticRegressionLoss(Matrix X, Vector y, Vector yHat)
        {
            //gradient of weights = X^T*(yHat-y)/m     m is sample number
            var gradientWeights = (X.Transpose() * (yHat - y)) / y.Length;
            //biasGradient = (yHat-y)/SampleNumber
            var gradientBias = (yHat - y).Mean();
            return (gradientWeights.ToVector(), gradientBias);
        }

        public static (Matrix, Vector) SoftmaxLoss(Matrix X, Matrix y, Matrix yHat)
        {
            var error = yHat - y;
            var gradientBias = new Vector(y.Columns);
            for (int i = 0; i < y.Columns; i++)
            {
                gradientBias[i] = error.GetColumn(i).Mean();
            }

            var mat = new Matrix(X.Columns, gradientBias.Length);
            for (int i = 0; i < X.Rows; i++)
            {
                var m = X[i].ToMatrix(true) * gradientBias.ToMatrix();
                mat += m;
            }
            var gradientWeights = mat / X.Rows;

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

        #endregion

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
