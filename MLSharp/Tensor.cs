using System;
using System.Collections.Generic;
using System.Text;


namespace MLSharp
{
    public static class Tensor
    {
        public static double[] GetRow(double[,] matrix, int index)
        {
            var columns = matrix.GetLength(1);
            var result = new double[columns];
            for (int i = 0; i < columns; i++)
            {
                result[i] = matrix[index, i];
            }

            return result;
        }

        public static double[] GetColumn(double[,] matrix, int index)
        {
            var rows = matrix.GetLength(0);
            var result = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                result[i] = matrix[i, index];
            }

            return result;
        }

        public static double Sum(double[] vector)
        {
            var result = 0d;
            for (int i = 0; i < vector.Length; i++)
            {
                result += vector[i];
            }
            return result;
        }

        public static double Sum(double[,] matrix)
        {
            var result = 0d;
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    result += matrix[i,j];
                }
            }
            return result;
        }

        public static double[] Add(double[] vector, double b)
        {
            var len = vector.Length;
            var result = new double[len];
            for (int i = 0; i < len; i++)
            {
                result[i] = vector[i] + b;
            }

            return result;
        }

        public static double[,] Add(double[,] matrix, double b)
        {
            var rows = matrix.GetLength(0);
            var columns = matrix.GetLength(1);

            var result = new double[rows, columns];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    result[i, j] = matrix[i, j] + b;
                }
            }

            return result;
        }

        public static double[] Add(double[] vector1, double[] vector2)
        {
            if (vector1.Length != vector2.Length)
                throw new Exception("vector1 and vector2 are not the same length!");

            var result = new double[vector1.Length];

            for (int i = 0; i < vector1.Length; i++)
            {
                result[i] = vector1[i] + vector2[i];
            }

            return result;
        }

        public static double[,] Add(double[,] matrix1, double[,] matrix2)
        {
            if (matrix1.GetLength(0) != matrix2.GetLength(0) || matrix1.GetLength(1) != matrix2.GetLength(1))
                throw new Exception("matrix1 and matrix2 are not the same shape!");

            var rows = matrix1.GetLength(0);
            var columns = matrix2.GetLength(1);

            var result = new double[rows, columns];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    result[i, j] = matrix1[i, j] + matrix2[i, j];
                }
            }

            return result;
        }

        public static double[] Multiple(double[] vector, double b)
        {
            var len = vector.Length;
            var result = new double[len];
            for (int i = 0; i < len; i++)
            {
                result[i] = vector[i] * b;
            }

            return result;
        }
    }
}
