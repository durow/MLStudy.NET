/* 
 * Description:This class is used to generate test data.
 * Author:AYX
 * date:2018.10.19
 */

using System;
using System.Collections.Generic;

namespace MLStudy.Num
{
    public class DataGenerator
    {
        private static DataGenerator instance;
        public static DataGenerator Instance
        {
            get
            {
                if (instance == null)
                    instance = new DataGenerator();
                return instance;
            }

        }

        private static Random rand1 = new Random();
        private static Random rand2 = new Random();

        public double Random()
        {
            return rand1.NextDouble();
        }

        public float[] RandFloat(int length)
        {
            var result = new float[length];
            for (int i = 0; i < length; i++)
            {
                result[i] = (float)rand1.NextDouble();
            }
            return result;
        }

        public int[] RandInt(int length)
        {
            var result = new int[length];
            for (int i = 0; i < length; i++)
            {
                result[i] = rand1.Next();
            }
            return result;
        }

        public double[] RandDouble(int length)
        {
            var result = new double[length];
            for (int i = 0; i < length; i++)
            {
                result[i] = rand1.NextDouble();
            }
            return result;
        }

        public float[] RandGaussianFloat(int length, double mean = 0, double variance = 1)
        {
            var len = length / 2 + length % 2;
            var result = new float[length];

            for (int i = 0; i < len; i++)
            {
                var u = rand1.NextDouble();
                var v = rand2.NextDouble();
                var randg = RandomGaussian(u, v, mean, variance);
                result[i * 2] = (float)randg[0];
                if (length > i * 2 + 1)
                    result[i * 2 + 1] = (float)randg[1];
            }
            return result;
        }

        public double[] RandGaussianDouble(int length, double mean = 0, double variance = 1)
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
