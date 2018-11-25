using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class Derivatives
    {
        public static double ReLU(double d)
        {
            return d > 0 ? 1 : 0;
        }

        public static double Tanh(double d)
        {
            var g = Functions.Tanh(d);
            return 1 - Math.Pow(g, 2);
        }

        public static double TanhFromOutput(double output)
        {
            return 1 - Math.Pow(output, 2);
        }

        public static double Sigmoid(double input)
        {
            var o = Functions.Sigmoid(input);
            return o * (1 - o);
        }

        public static double SigmoidFromOutput(double output)
        {
            return output * (1 - output);
        }

        public static double[,] Softmax(double[] x)
        {
            var output = Functions.Softmax(x);
            return SoftmaxFromOutput(output);
        }

        public static double[,] SoftmaxFromOutput(double[] output)
        {
            var len = output.Length;
            var jacob = new double[len, len];
            for (int i = 0; i < len; i++)
            {
                for (int j = 0; j < len; j++)
                {
                    if (i == j)
                        jacob[i, j] = output[i] * (1 - output[j]);
                    else
                        jacob[i, j] = -output[i] * output[j];
                }
            }
            return jacob;
        }

        public static double[] CrossEntropy(double[] y, double[] yHat)
        {
            if (y.Length != yHat.Length)
                throw new Exception("y and yhat must be the same length");

            var result = new double[y.Length];
            CrossEntropy(y, yHat, result);
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
                return -1 / yHat;

            if (y == 0) 
                return 1 / (1 - yHat);

            return -y / yHat; //这个返回实际上是无意义的
        }

        public static void CrossEntropy(double[] y, double[] yHat, double[] result)
        {
            if(result.Length == 1)
            {
                result[0] = CrossEntropy(y[0], yHat[0]);
                return;
            }

            for (int i = 0; i < result.Length; i++)
            {
                if (y[i] == 0)
                    result[i] = 0;
                else
                    result[i] = -y[i] / yHat[i];
            }
        }

        public static Tensor MeanSquareError(Tensor y, Tensor yHat)
        {
            Tensor.CheckShape(y, yHat);

            var result = y.GetSameShape();
            MeanSquareError(y, yHat, result);
            return result;
        }

        public static void MeanSquareError(Tensor y, Tensor yHat, Tensor result)
        {
            //因为存在learning rate，所以梯度前面的系数不那么重要，但最好和损失函数一致，
            Tensor.Minus(yHat, y, result);
            result.Multiple(2d / y.ElementCount);
        }
    }
}
