using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

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

        public static Vector Softmax(Vector v)
        {
            var max = v.Max();
            v -= max;
            v = v.ApplyFunction(a => Math.Pow(Math.E, a));
            return v / v.Sum();
        }

        public static Matrix SoftmaxByRow(Matrix m)
        {
            var result = m.GetSameShape();
            for (int i = 0; i < m.Rows; i++)
            {
                var s = Softmax(m[i]);
                for (int j = 0; j < m.Columns; j++)
                {
                    result[i, j] = s[j];
                }
            }
            return result;
        }
    }
}
