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
            var (gradientWeights, gradientBias) = Gradient.LogisticRegressionLoss(X, y, LastYHat);
            gradientWeights += regularizationGradient[Regularization](Weights, RegularizationWeight);

            Weights -= LearningRate * gradientWeights;
            Bias -= LearningRate * gradientBias;
        }

        public override double Loss(Matrix X, Vector y)
        {
            var yHat = Predict(X);

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

        public override double Error(Vector yHat, Vector y)
        {
            yHat = (yHat - 0.5).ApplyFunction(Functions.IndicatorFunction);
            return LossFunctions.ErrorPercent(yHat, y); 
        }
    }
}
