using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Deep
{
    public class CrossEntropy : ILossFunction
    {
        public Tensor GetGradient(Tensor y, Tensor yHat)
        {
            Tensor.CheckShape(y, yHat);

            return Tensor.DivideElementWise(y, yHat).Multiple(-1);
        }

        public double GetLoss(Tensor y, Tensor yHat)
        {
            Tensor.CheckShape(y, yHat);
            if (y.Rank != 2)
                throw new TensorShapeException("to compute cross entropy, Tensor.Rank must be 2!");

            var sampleCount = y.Shape[0];
            var result = 0d;
            for (int i = 0; i < sampleCount; i++)
            {
                result += Function(y.GetTensorByDim1(i).GetRawValues(), yHat.GetTensorByDim1(i).GetRawValues());
            }
            return result / sampleCount;
        }

        public static double Function(double[] y, double[] yHat)
        {
            if (y.Length != yHat.Length)
                throw new Exception("y and yhat must be the same length");

            if (y.Length == 1)
                return Function(y[0], yHat[0]);

            var result = 0d;
            for (int i = 0; i < y.Length; i++)
            {
                if (y[i] == 0)
                    continue;

                result -= y[i] * Math.Log(yHat[i]);
            }

            return result;
        }

        public static double Function(double y, double yHat)
        {
            if (y == 1)
                return -Math.Log(yHat);

            if (y == 0)
                return -Math.Log(1 - yHat);

            return -y * Math.Log(yHat) - (1 - y) * Math.Log(1 - yHat);
        }

        public static double[] Derivative(double[] y, double[] yHat)
        {
            if (y.Length != yHat.Length)
                throw new Exception("y and yhat must be the same length");

            var result = new double[y.Length];
            for (int i = 0; i < y.Length; i++)
            {
                result[i] = -y[i] / yHat[i];
            }
            return result;
        }
    }
}
