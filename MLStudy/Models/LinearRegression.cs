using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class LinearRegression
    {
        public Vector Weights { get; private set; }
        public double Bias { get; private set; }
        public double LearningRate { get; set; } = 0.01;
        public LinearRegularization Regularization { get; set; } = LinearRegularization.None;

        public void Train(Matrix X, Vector y)
        { }

        public void Step(Matrix X, Vector y)
        {
            var yHat = Predict(X);
            var weightsGradient = Gradient.LinearWeights(X, yHat, y);
            var biasGradient = Gradient.LinearBias(yHat, y);
            Weights -= LearningRate * weightsGradient;
            Bias -= LearningRate * biasGradient;
        }

        public Vector Predict(Matrix X)
        {
            var result = X * Weights + Bias;
            return result.GetColumn(1);
        }

        public double Predict(Vector x)
        {
            return 0;
        }
    }

    public class LinearStepInfo
    {
        public Vector OldWeights { get; internal set; }
        public Vector NewWeights { get; internal set; }
        public double OldBias { get; internal set; }
        public double NewBias { get; internal set; }
        public Vector Y { get; set; }
        public Vector YHat { get; set; }
        public double Error { get; set; }
    }

    public enum LinearRegularization
    {
        None,
        L1,
        L2
    }
}
