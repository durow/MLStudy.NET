/* 
 * Description:This class is used to generate test data.
 * Author:AYX
 * date:2018.10.19
 */

using System;
using System.Collections.Generic;

namespace MLStudy
{
    public class DataEmulator
    {
        private static DataEmulator instance;
        public static DataEmulator Instance
        {
            get
            {
                if (instance == null)
                    instance = new DataEmulator();
                return instance;
            }

        }

        private static Random rand1 = new Random((int)DateTime.Now.Ticks);
        private static Random rand2 = new Random((int)DateTime.Now.Ticks);

        public (Matrix,Vector,Matrix,Vector) GetTrainTestData(int rows, int columns, int testSize, double min, double max, Func<Matrix,Vector> F)
        {
            Matrix trainX, testX;
            Vector trainY, testY;
            (trainX, trainY) = GetTrainData(rows, columns, min, max, F);
            (testX, testY) = GetTrainData(testSize, columns, min, max, F);

            return (trainX, trainY, testX, testY);
        }

        public (Matrix, Vector) GetTrainData(int rows, int columns, double min, double max, Func<Matrix, Vector> F)
        {
            var trainX = RandomMatrix(rows, columns) * (max - min) + min;
            var trainY = F(trainX);
            return (trainX, trainY);
        }

        public double Random()
        {
            return rand1.NextDouble();
        }

        public double[] RandomArray(int length)
        {
            var result = new double[length];
            for (int i = 0; i < length; i++)
            {
                result[i] = rand1.NextDouble();
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

        public List<double[]> RandomList(int rows, int columns)
        {
            var len = rows * columns;
            var result = new double[rows, columns];
            var rand = RandomArray(len);
            return D1ToList(rand, rows, columns);
        }

        public double[] RandomArrayGaussian(int length, double mean = 0, double variance = 1)
        {
            var len = length / 2 + length % 2;
            var result = new double[length];

            for (int i = 0; i < len; i++)
            {
                var u = rand1.NextDouble();
                var v = rand2.NextDouble();
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

        public Vector RandomVector(int length, double min, double max)
        {
            return RandomVector(length) * (max - min) + min;
        }

        public Matrix RandomMatrix(int rows, int columns)
        {
            return new Matrix(RandomArray(rows,columns));
        }

        public Matrix RandomMatrix(int rows, int columns, double min, double max)
        {
            return RandomMatrix(rows, columns) * (max - min) + min;
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

        private List<double[]> D1ToList(double[] data, int rows, int columns)
        {
            if (data.Length != rows * columns)
                throw new Exception("can't change!");

            var result = new List<double[]>(rows);
            var counter = 0;

            for (int i = 0; i < rows; i++)
            {
                var row = new double[columns];
                for (int j = 0; j < columns; j++)
                {
                    row[j] = data[counter];
                    counter++;
                }
                result.Add(row);
            }
            return result;
        }
    }
}
