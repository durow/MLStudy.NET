using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class LinearRegression:ITrainable
    {
        public Vector Weights { get; protected set; } = new Vector();
        public double Bias { get; protected set; } = 1;
        public double LearningRate
        {
            get
            {
                return Optimizer.LearningRate;
            }
            set
            {
                Optimizer.LearningRate = value;
            }
        }
        public GradientOptimizer Optimizer { get; private set; } = new GradientOptimizer();
        public double RegularizationWeight
        {
            get
            {
                return Regularization.RegularizationWeight;
            }
            set
            {
                Regularization.RegularizationWeight = value;
            }
        }
        public RegularTypes RegularizationType
        {
            get
            {
                return Regularization.RegularType;
            }
            set
            {
                Regularization.RegularType = value;
            }
        }
        public Regularization Regularization { get; private set; } = new Regularization();

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

            var yHat = Predict(X);
            var (gradientWeights, gradientBias) = Gradient.LinearSquareError(X, y, yHat);

            if (Regularization.RegularType != RegularTypes.None)
                gradientWeights += Regularization.GetValue(Weights);

            Weights = Optimizer.GradientDescent(Weights, gradientWeights);
            Bias = Optimizer.GradientDescent(Bias, gradientBias);
        }

        public Vector Predict(Matrix X)
        {
            var result = X * Weights + Bias;
            return result.ToVector();
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
}
