using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class LinearRegressionLayer
    {
        public Matrix Weights { get; private set; }
        public double Bias { get; private set; }
        public int InputFeatures { get; private set; }

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

        public GradientOptimizer Optimizer { get; set; } = new GradientOptimizer();

        public Matrix ForwardInput { get; protected set; }
        public Matrix ForwardOutput { get; protected set; }
        public Matrix InputError { get; protected set; }
        public Matrix LinearError { get; protected set; }
        public double Loss { get; protected set; }

        public LinearRegressionLayer(int inputFeatures)
        {
            InputFeatures = inputFeatures;
            AutoInitWeightBias();
        }

        public virtual void AutoInitWeightBias()
        {
            Weights = new Matrix(InputFeatures, 1);
            Bias = 1;
        }

        public virtual Matrix Backward(Vector y)
        {
            ComputeOutputError(y);
            ErrorBP(y);
            UpdateWeightsBias();
            return InputError;
        }

        protected virtual void ComputeOutputError(Vector y)
        {
            var yHat = ForwardOutput.ToVector();
            Loss = LossFunctions.MeanSquareError(yHat, y);
        }

        protected virtual void ErrorBP(Vector y)
        {
            var matrixY = y.ToMatrix(true);
            LinearError = ForwardOutput - matrixY;
            InputError = LinearError * Weights.Transpose();
        }

        protected virtual void UpdateWeightsBias()
        {
            var weightsGradient = ForwardInput.Transpose() * LinearError / LinearError.Rows;
            var biasGradient = LinearError.Mean();
            Weights = Optimizer.GradientDescent(Weights, weightsGradient);
            Bias = Optimizer.GradientDescent(Bias, biasGradient);
        }

        public virtual Matrix Forward(Matrix input)
        {
            ForwardInput = input;
            ForwardOutput = input * Weights + Bias;
            return ForwardOutput;
        }
    }
}
