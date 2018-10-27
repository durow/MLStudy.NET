using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class LinearRegression:ITrain
    {
        public Vector Weights { get; protected set; } = new Vector();
        public double Bias { get; protected set; } = 1;
        public double LearningRate { get; set; } = 0.0001;
        public LinearRegularization Regularization { get; set; } = LinearRegularization.None;
        public double RegularizationWeight { get; set; } = 0.01;
        protected Dictionary<LinearRegularization, Func<Vector, double, Vector>> regularizationGradient = new Dictionary<LinearRegularization, Func<Vector, double, Vector>>();

        public int StepCounter { get; private set; } = 0;
        public Vector LastYHat { get; protected set; }

        public LinearRegression()
        {
            regularizationGradient.Add(LinearRegularization.L1, Gradient.LinearL1);
            regularizationGradient.Add(LinearRegularization.L2, Gradient.LinearL2);
            regularizationGradient.Add(LinearRegularization.None, (a,b)=>new Vector(a.Length));
        }

        public void SetWeights(params double[] weights)
        {
            Weights = new Vector(weights);
        }

        public void SetBias(double bias)
        {
            Bias = bias;
        }

        public virtual void Step(Matrix X, Vector y)
        {
            if(Weights.Length != X.Columns)
            {
                AutoInitWeight(X.Columns);
            }

            LastYHat = Predict(X);
            var (gradientWeights, gradientBias) = Gradient.LinearRegressionLoss(X, y, LastYHat);
            gradientWeights += regularizationGradient[Regularization](Weights, RegularizationWeight);

            Weights -= LearningRate * gradientWeights;
            Bias -= LearningRate * gradientBias;

            StepCounter++;
        }

        public Vector Predict(Matrix X)
        {
            var result = X * Weights + Bias;
            return result.ToVector();
        }

        public double Predict(Vector x)
        {
            return 0;
        }

        public virtual double Loss(Matrix X, Vector y)
        {
            var yHat = Predict(X);
            return LossFunctions.MeanSquareError(yHat, y);
        }

        public double Error(Vector yHat, Vector y)
        {
            return LossFunctions.MeanSquareError(yHat, y);
        }

        protected void AutoInitWeight(int length)
        {
            var emu = new DataEmulator();
            Weights = emu.RandomVector(length);
        }

    }

    public enum LinearRegularization
    {
        None,
        L1,
        L2
    }
}
