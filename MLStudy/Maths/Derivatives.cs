using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class Derivatives
    {
        public static Matrix ReLU(Matrix input)
        {
            return input.ApplyFunction(a => ReLU(a));
        }

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
    }
}
