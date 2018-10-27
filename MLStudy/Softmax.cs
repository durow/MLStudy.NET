using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class Softmax : IMachine
    {
        public double LearningRate { get; set; }
        public Matrix Weights { get; private set; }
        public Vector Bias { get; set; }
        public int Types { get; set; }

        public Softmax(int types)
        {
            Types = types;
        }

        public double Error(Vector yHat, Vector y)
        {
            return LossFunctions.ErrorPercent(yHat, y);
        }

        public double Loss(Vector yHat, Vector y)
        {
            return LossFunctions.CrossEntropy(y, yHat);
        }

        public double Predict(Vector x)
        {
            var z = (x * Weights).ToVector() + Bias;
            var sm = Functions.Softmax(z);
            return FindClass(sm);
        }

        public Vector Predict(Matrix X)
        {
            var z = Bias + X * Weights;
            var sm = Functions.SoftmaxByRow(z);
            return FindClass(sm);
        }

        public void Step(Matrix X, Vector y)
        {
            var matrixY = ExtendToMatrix(y);
            var z = Bias + X * Weights;
            var yHat = Functions.SoftmaxByRow(z);
            var (gradientWeights, gradientBias) = Gradient.SoftmaxLoss(X, matrixY, yHat);
            Weights -= gradientWeights * LearningRate;
            Bias -= gradientBias * LearningRate;
        }

        private Matrix ExtendToMatrix(Vector v)
        {
            var result = new Matrix(v.Length, Types);
            for (int i = 0; i < v.Length; i++)
            {
                for (int j = 0; j < Types; j++)
                {
                    if (j == v[i])
                        result[i, j] = 1;
                    else
                        result[i, j] = 0;
                }
            }
            return result;
        }

        private Vector FindClass(Matrix m)
        {
            var result = new Vector(m.Rows);
            for (int i = 0; i < m.Rows; i++)
            {
                result[i] = FindClass(m[i]);
            }
            return result;
        }

        private double FindClass(Vector v)
        {
            var c = 0;
            var p = 0d;

            for (int i = 0; i < v.Length; i++)
            {
                if(v[i] > p)
                {
                    c = i;
                    p = v[i];
                }
            }

            return c;
        }
    }
}
