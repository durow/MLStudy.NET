using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public static class Functions
    {
        public static double IndicatorFunction(double d)
        {
            return d >= 0 ? 1 : 0;
        }

        public static double Sigmoid(double x)
        {
            return 1 / (1 + Math.Pow(System.Math.E, -x));
        }

        public static Vector Sigmoid(Vector v)
        {
            var len = v.Length;
            var result = new double[len];
            for (int i = 0; i < len; i++)
            {
                result[i] = Sigmoid(v[i]);
            }
            return new Vector(result);
        }

        public static Matrix Sigmoid(Matrix m)
        {
            var result = new double[m.Rows, m.Columns];
            for (int i = 0; i < m.Rows; i++)
            {
                for (int j = 0; j < m.Columns; j++)
                {
                    result[i, j] = Sigmoid(m[i, j]);
                }
            }
            return new Matrix(result);
        }

        public static double ReLU(double x)
        {
            return Math.Max(0, x);
        }
    }
}
