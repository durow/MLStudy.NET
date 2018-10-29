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

        public Matrix ForwardInput { get; private set; }
        public Matrix ForwardOutput { get; private set; }
        public Matrix InputError { get; private set; }
        public Matrix LinearError { get; private set; }
        public double OutputError { get; private set; }

        public LinearRegressionLayer(int inputFeatures)
        {
            InputFeatures = inputFeatures;
            AutoInitWeightBias();
        }

        public void AutoInitWeightBias()
        {
            Weights = new Matrix(InputFeatures, 1);
            Bias = 1;
        }

        public Matrix Backward(Vector y)
        {
            ComputeOutputError(y);
            ErrorBP(y);
            UpdateWeightsBias();
            return InputError;
        }

        private void ComputeOutputError(Vector y)
        {
            var yHat = ForwardOutput.ToVector();
            OutputError = LossFunctions.MeanSquareError(yHat, y);
        }

        private void ErrorBP(Vector y)
        {
            var matrixY = y.ToMatrix(true);
            LinearError = ForwardOutput - matrixY;
            InputError = LinearError * Weights.Transpose();
        }

        private void UpdateWeightsBias()
        {
            var weightsGradient = ForwardInput.Transpose() * LinearError / LinearError.Rows;
            var biasGradient = LinearError.Mean();
            Weights = Optimizer.GradientDescent(Weights, weightsGradient);
            Bias = Optimizer.GradientDescent(Bias, biasGradient);
        }

        public Matrix Forward(Matrix input)
        {
            ForwardOutput = input * Weights + Bias;
            return ForwardOutput;
        }
    }
}
