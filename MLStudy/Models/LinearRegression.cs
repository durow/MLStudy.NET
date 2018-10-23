using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class LinearRegression:IMachine
    {
        public Vector Weights { get; private set; } = new Vector(0);
        public double Bias { get; private set; }
        public double LearningRate { get; set; } = 0.0001;
        public LinearRegularization Regularization { get; set; } = LinearRegularization.None;
        public double RegularizationWeight { get; set; } = 0.01;
        private Dictionary<LinearRegularization, Func<Vector, double, Vector>> regularizationGradient = new Dictionary<LinearRegularization, Func<Vector, double, Vector>>();

        public int StepCounter { get; private set; } = 0;
        public Vector LastYHat { get; private set; }

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

        public void Step(Matrix X, Vector y)
        {
            if(Weights.Length != X.Columns)
            {
                AutoInitWeight(X.Columns);
            }

            LastYHat = Predict(X);
            var weightsGradient = Gradient.LinearWeights(X, LastYHat, y) + regularizationGradient[Regularization](Weights, RegularizationWeight);
            var biasGradient = Gradient.LinearBias(LastYHat, y);

            Weights -= LearningRate * weightsGradient;
            Bias -= LearningRate * biasGradient;

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

        public void ResetStepCounter()
        {
            StepCounter = 0;
        }

        private void AutoInitWeight(int length)
        {
            var emu = new DataEmulator();
            Weights = emu.RandomVector(length);
        }
    }

    public class LinearStepInfo
    {
        public int StepCounter { get; internal set; }
        public Vector OldWeights { get; internal set; }
        public Vector NewWeights { get; internal set; }
        public double OldBias { get; internal set; }
        public double NewBias { get; internal set; }
        public Vector Y { get; internal set; }
        public Vector YHat { get; internal set; }
        public double Error { get; internal set; }
    }

    public enum LinearRegularization
    {
        None,
        L1,
        L2
    }
}
