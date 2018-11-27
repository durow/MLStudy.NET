using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using ILNumerics;

namespace MLStudy
{
    public static class Functions
    {
        public static double Indicator(double d)
        {
            return d >= 0 ? 1 : 0;
        }

        public static double Sigmoid(double x)
        {

            return 1 / (1 + Math.Exp(-x));
        }

        public static double Tanh(double x)
        {
            var pos = Math.Exp(x);
            var neg = Math.Exp(-x);

            return (pos - neg) / (pos + neg);
        }

        public static double ReLU(double x)
        {
            return Math.Max(0, x);
        }

        public static double[] Softmax(double[] x)
        {
            var result = new double[x.Length];
            Softmax(x, result);
            return result;
        }

        public static void Softmax(double[] x, double[] result)
        {
            var max = x.Max();
            for (int i = 0; i < x.Length; i++)
            {
                result[i] = Math.Exp(x[i] - max);
            }

            var sum = result.Sum();
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = result[i] / sum;
            }
        }

        public static double CrossEntropy(double[] y, double[] yHat)
        {
            if (y.Length != yHat.Length)
                throw new Exception("y and yhat must be the same length");

            if (y.Length == 1)
                return CrossEntropy(y[0], yHat[0]);

            var result = 0d;
            for (int i = 0; i < y.Length; i++)
            {
                if (y[i] == 0)
                    continue;

                result -= y[i] * Math.Log(yHat[i]);
            }

            return result;
        }

        /// <summary>
        /// 这个方法主要用于二分类问题，只求得正分类概率的情况下
        /// 例如使用Sigmoid输出
        /// </summary>
        /// <param name="y">真实值</param>
        /// <param name="yHat">预测值</param>
        /// <returns>交叉熵</returns>
        public static double CrossEntropy(double y, double yHat)
        {
            if (y == 1)
                return -Math.Log(yHat);

            if (y == 0)
                return -Math.Log(1 - yHat);

            return -y * Math.Log(yHat) - (1 - y) * Math.Log(1 - yHat);
        }

        public static double MeanSquareError(Tensor y, Tensor yHat)
        {
            return SquareError(y, yHat).Mean();
        }

        public static Tensor SquareError(Tensor y, Tensor yHat)
        {
            Tensor.CheckShape(y, yHat);

            var result = y.GetSameShape();
            SquareError(y, yHat, result);
            return result;
        }

        public static void SquareError(Tensor y, Tensor yHat, Tensor result)
        {
            Tensor.Minus(y, yHat, result);
            result.Apply(a => a * a);
        }
    }
}
