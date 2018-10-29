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

        public new LogisticResult Predict(Vector x)
        {
            var a = Tensor.MultipleAsMatrix(x, Weights) + Bias;
            var raw = Functions.Sigmoid(a);
            var result = Functions.IndicatorFunction(raw - 0.5);
            return new LogisticResult(raw, result);
        }

        public new LogisticResultMulti Predict(Matrix X)
        {
            var a = X * Weights + Bias;
            var raw = a.ToVector().ApplyFunction(Functions.Sigmoid);
            var result = ProbabilityToCategory(raw);
            return new LogisticResultMulti(raw, result);
        }

        public override void Step(Matrix X, Vector y)
        {
            if (Weights.Length != X.Columns)
            {
                AutoInitWeight(X.Columns);
            }

            LastYHat = Predict(X).RawResult;
            var (gradientWeights, gradientBias) = Gradient.LinearSigmoidCrossEntropy(X, y, LastYHat);
            gradientWeights += regularizationGradient[Regularization](Weights, RegularizationWeight);

            Weights -= LearningRate * gradientWeights;
            Bias -= LearningRate * gradientBias;
        }

        public override double Loss(Matrix X, Vector y)
        {
            var yHat = Predict(X).RawResult;
            return LossFunctions.LogisticError(yHat, y);
        }

        public double Error(Matrix X, Vector y)
        {
            var yHat = Predict(X).Result;
            return LossFunctions.ErrorPercent(yHat, y); 
        }

        private Vector ProbabilityToCategory(Vector v)
        {
            return (v - 0.5).ApplyFunction(Functions.IndicatorFunction);
        }
    }

    public class LogisticResult
    {
        public double Result { get; private set; }
        public double RawResult { get; private set; }

        public LogisticResult(double raw, double result)
        {
            RawResult = raw;
            Result = result;
        }
    }

    public class LogisticResultMulti
    {
        public Vector Result { get; private set; }
        public Vector RawResult { get; private set; }

        public LogisticResultMulti(Vector raw, Vector result)
        {
            RawResult = raw;
            Result = result;
        }
    }
}
