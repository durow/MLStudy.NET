﻿using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class LinearRegressionOut: OutputLayer
    {
        public Matrix Weights { get; set; }
        public double Bias { get; set; }

        public LinearRegressionOut(int inputFeatures)
        {
            InputFeatures = inputFeatures;
        }

        public override void AutoInitWeightsBias()
        {
            Weights = new Matrix(InputFeatures, 1);
            Bias = 1;
        }

        public override Matrix Forward(Matrix input)
        {
            ForwardInput = input;
            ForwardOutput = input * Weights + Bias;
            return ForwardOutput;
        }

        public override Matrix Backward(Vector y)
        {
            GetLoss(y);
            ErrorBP(y);
            UpdateWeightsBias();
            return InputError;
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

        public override double GetLoss(Matrix yHat, Vector y)
        {
            var v = yHat.ToVector();
            return LossFunctions.MeanSquareError(v, y);
        }

        public override double GetError(Vector y)
        {
            return LossFunctions.MeanSquareError(ForwardOutput.ToVector(), y);
        }

        public override Vector GetPredict()
        {
            return ForwardOutput.ToVector();
        }
    }
}