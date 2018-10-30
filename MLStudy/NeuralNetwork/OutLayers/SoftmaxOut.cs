using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class SoftmaxOut: OutputLayer
    {
        public Matrix Weights { get; private set; }
        public Vector Bias { get; set; }
        public int CategoryCount { get; private set; }

        public SoftmaxOut(int inputeFeatures, int categoryCount)
        {
            InputFeatures = inputeFeatures;
            CategoryCount = categoryCount;
            AutoInitWeightsBias();
        }

        public override void AutoInitWeightsBias()
        {
            Weights = new Matrix(InputFeatures, CategoryCount);
            Bias = new Vector(CategoryCount);
        }

        public override Matrix Forward(Matrix input)
        {
            ForwardInput = input;
            var z = Bias + input * Weights;
            ForwardOutput = Functions.SoftmaxByRow(z);
            return ForwardOutput;
        }

        public override Matrix Backward(Vector y)
        {
            GetLoss(y);
            var matrixY = ExtendSoftmaxResultToMatrix(y);
            ErrorBP(matrixY);
            UpdateWeightsBias();
            throw new NotImplementedException();
        }

        private void ErrorBP(Matrix y)
        {
            LinearError = ForwardOutput - y;
            InputError = LinearError * Weights.Transpose();
        }

        private void UpdateWeightsBias()
        {
            var v = new Vector(LinearError.Columns, 1);
            var gradientBias = (v * LinearError).ToVector() / LinearError.Rows;
            var gradientWeights = ForwardInput.Transpose() * LinearError / LinearError.Rows;

            Weights = Optimizer.GradientDescent(Weights, gradientWeights);
            Bias = Optimizer.GradientDescent(Bias, gradientBias);
        }

        public override double GetLoss(Matrix yHat, Vector y)
        {
            var matrixY = ExtendSoftmaxResultToMatrix(y);
            var sum = 0d;
            for (int i = 0; i < ForwardOutput.Rows; i++)
            {
                var ce = LossFunctions.CrossEntropy(ForwardOutput[i], matrixY[i]);
                sum += ce;
            }
            return sum / y.Length;
        }

        private Matrix ExtendSoftmaxResultToMatrix(Vector v)
        {
            var result = new Matrix(v.Length, CategoryCount);
            for (int i = 0; i < v.Length; i++)
            {
                for (int j = 0; j < CategoryCount; j++)
                {
                    if (j == v[i])
                        result[i, j] = 1;
                    else
                        result[i, j] = 0;
                }
            }
            return result;
        }

        
    }
}
