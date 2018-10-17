using System;
using System.Collections.Generic;
using System.Text;

namespace MLSharp
{
    public static class Functions
    {
        public static double Sigmoid(double x)
        {
            return 1 / (1 + System.Math.Pow(System.Math.E, -x));
        }

        public static double[] Sigmoid(double[] vector)
        {
            var len = vector.Length;
            var result = new double[len];
            for (int i = 0; i < len; i++)
            {
                result[i] = Sigmoid(vector[i]);
            }
            return result;
        }
    }
}
