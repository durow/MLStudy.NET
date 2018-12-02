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

        private static Random rand1 = new Random();
        private static Random rand2 = new Random();

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
