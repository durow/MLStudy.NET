using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace MLStudy
{
    public class DataEmulator
    {
        public double[] RandomArray(int length)
        {
            var rand = new Random((int)DateTime.Now.Ticks);
            var result = new double[length];
            for (int i = 0; i < length; i++)
            {
                result[i] = rand.NextDouble();
            }
            return result;
        }

        public double[,] RandomArray(int rows, int columns)
        {
            var len = rows * columns;
            var result = new double[rows, columns];
            var rand = RandomArray(len);
            return D1ToD2(rand, rows, columns);
        }

        public double[] RandomArrayGaussian(int length, double mean = 0, double variance = 1)
        {
            var len = length / 2 + length % 2;
            var result = new double[length];
            var r1 = new Random((int)DateTime.Now.Ticks);
            var r2 = new Random(DateTime.Now.Second);
            for (int i = 0; i < len; i++)
            {
                var u = r1.NextDouble();
                var v = r2.NextDouble();
                var randg = RandomGaussian(u, v, mean, variance);
                result[i * 2] = randg[0];
                if (length > i * 2 + 1)
                    result[i * 2 + 1] = randg[1];
            }
            return result;
        }

        public double[,] RandomArrayGaussian(int rows, int columns, double mean = 0, double variance = 1)
        {
            var len = rows * columns;
            var result = new double[rows, columns];
            var rand = RandomArrayGaussian(len);
            return D1ToD2(rand, rows, columns);
        }

        public Vector RandomVector(int length)
        {
            return new Vector(RandomArray(length));
        }

        public Matrix RandomMatrix(int rows, int columns)
        {
            return new Matrix(RandomArray(rows,columns));
        }

        public Vector RandomVectorGaussian(int length, double mean = 0, double variance = 1)
        {
            return new Vector(RandomArrayGaussian(length, mean, variance));
        }

        public Matrix RandomMatrixGaussian(int rows, int columns, double mean = 0, double variance = 1)
        {
            return new Matrix(RandomArrayGaussian(rows, columns, mean, variance));
        }

        public Vector AddGaussianNoise(Vector v, double mean = 0, double variance = 1)
        {
            var noise = RandomVectorGaussian(v.Length, mean, variance);
            return v + noise;
        }

        public Matrix AddGaussianNoise(Matrix m, double mean = 0, double variance = 1)
        {
            var noise = RandomMatrixGaussian(m.Rows, m.Columns, mean, variance);
            return m + noise;
        }

        //z0 = sqrt(-2 * log(u)) * cos(2*pi * v);
        //z1 = sqrt(-2 * log(u)) * sin(2*pi* v);
        //u and v are between (0,1)
        public double[] RandomGaussian(double u, double v, double mean, double variance)
        {
            var a = Math.Sqrt(-2 * Math.Log(u));
            var b = 2 * Math.PI * v;
            var z0 = a * Math.Cos(b) * variance + mean;
            var z1 = a * Math.Sin(b) * variance + mean;
            return new double[] { z0, z1 };
        }

        private double[,] D1ToD2(double[] data, int rows, int columns)
        {
            if (data.Length != rows * columns)
                throw new Exception("can't change!");

            var result = new double[rows, columns];

            var counter = 0;
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    result[i, j] = data[counter];
                    counter++;
                }
            }
            return result;
        }
    }
}
