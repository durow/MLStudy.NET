using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class LogisticRegression : LinearRegression
    {
        public LogisticRegression():base()
        {
        }

        public override double Predict(Vector x)
        {
            var a = Tensor.MultipleAsMatrix(x, Weights) + Bias;
            return Functions.Sigmoid(a);
        }

        public override Vector Predict(Matrix X)
        {
            var a = X * Weights + Bias;
            return a.ToVector().ApplyFunction(Functions.Sigmoid);
        }

        public override void Step(Matrix X, Vector y)
        {
            if (Weights.Length != X.Columns)
            {
                AutoInitWeight(X.Columns);
            }

            LastYHat = Predict(X);
            var (gradientWeights, gradientBias) = Gradient.LogisticRegression(X, y, LastYHat);
            gradientWeights += regularizationGradient[Regularization](Weights, RegularizationWeight);

            Weights -= LearningRate * gradientWeights;
            Bias -= LearningRate * gradientBias;
        }

        public override double Loss(Vector yHat, Vector y)
        {
            var sum = 0d;

            for (int i = 0; i < y.Length; i++)
            {
                var crossEntropy = -y[i] * Math.Log(yHat[i]) - (1 - y[i]) * Math.Log(1 - y[i]);
                sum += crossEntropy;
            }

            return sum / y.Length;
        }
    }
}
