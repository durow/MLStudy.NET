using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class Softmax : ITrain
    {
        public double LearningRate { get; set; }
        public Matrix Weights { get; private set; }
        public Vector Bias { get; set; }
        public int CategoryCount { get; set; }

        public Softmax(int categoryCount)
        {
            CategoryCount = categoryCount;
        }

        public double Error(Matrix X, Vector y)
        {
            var yHat = Predict(X).Result;
            return LossFunctions.ErrorPercent(yHat, y);
        }

        public double Loss(Matrix X, Vector y)
        {
            var pHat = Predict(X).RawResult;
            var p = ExtendToMatrix(y);

            var sum = 0d;
            for (int i = 0; i < X.Rows; i++)
            {
                var ce = LossFunctions.CrossEntropy(p[i], pHat[i]);
                sum += ce;
            }
            return sum / y.Length;
        }

        public SoftmaxResult Predict(Vector x)
        {
            var z = (x * Weights).ToVector() + Bias;
            var raw = Functions.Softmax(z);
            var result = ProbabilityToCategory(raw);
            return new SoftmaxResult(raw, result);
        }

        public SoftmaxResultMulti Predict(Matrix X)
        {
            var z = Bias + X * Weights;
            var raw = Functions.SoftmaxByRow(z);
            var result = ProbabilityToCategory(raw);
            return new SoftmaxResultMulti(raw, result);
        }

        public void Step(Matrix X, Vector y)
        {
            var matrixY = ExtendToMatrix(y);
            var z = Bias + X * Weights; //add Bias(a Vector) to Weights(a Matrix) row by row
            var yHat = Functions.SoftmaxByRow(z);
            var (gradientWeights, gradientBias) = Gradient.SoftmaxLoss(X, matrixY, yHat);
            Weights -= gradientWeights * LearningRate;
            Bias -= gradientBias * LearningRate;
        }

        private Matrix ExtendToMatrix(Vector v)
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

        private Vector ProbabilityToCategory(Matrix m)
        {
            var result = new Vector(m.Rows);
            for (int i = 0; i < m.Rows; i++)
            {
                result[i] = ProbabilityToCategory(m[i]);
            }
            return result;
        }

        private double ProbabilityToCategory(Vector v)
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

    public class SoftmaxResult
    {
        public Vector RawResult { get; private set; }
        public double Result { get; private set; }

        public SoftmaxResult(Vector raw, double result)
        {
            RawResult = raw;
            Result = result;
        }
    }

    public class SoftmaxResultMulti
    {
        public Matrix RawResult { get; private set; }
        public Vector Result { get; private set; }

        public SoftmaxResultMulti(Matrix raw, Vector result)
        {
            RawResult = raw;
            Result = result;
        }
    }
}
