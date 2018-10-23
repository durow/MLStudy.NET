using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class LogisticRegression : IMachine
    {
        public double LearningRate { get; set; }
        public Vector Weights { get; private set; }
        public double Bias { get; private set; }


        public double Predict(Vector x)
        {
            var a = Tensor.MultipleAsMatrix(x, Weights) + Bias;
            return Functions.Sigmoid(a);
        }

        public Vector Predict(Matrix X)
        {
            var a = X * Weights + Bias;
            return a.ToVector().ApplyFunction(Functions.Sigmoid);
        }

        public void Step(Matrix X, Vector y)
        {
            var yHat = Predict(X);
        }

        public void SetWeight(Vector weights)
        {
            Weights = weights;
        }

        public void SetBias(double b)
        {
            Bias = b;
        }
    }
}
