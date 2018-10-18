using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;


namespace MLStudy
{
    public class Tensor
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

        #region Vector Operations

        public static Vector Add(Vector v, double b)
        {
            var result = new double[v.Length];
            for (int i = 0; i < v.Length; i++)
            {
                result[i] = v[i] + b;
            }
            return new Vector(result);
        }

        public static Vector Add(Vector a, Vector b)
        {
            CheckVectorLength(a, b);

            var result = new double[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                result[i] = a[i] + b[i];
            }
            return new Vector(result);
        }

        public static Vector Multiple(Vector v, double b)
        {
            var result = new double[v.Length];
            for (int i = 0; i < v.Length; i++)
            {
                result[i] = v[i] * b;
            }
            return new Vector(result);
        }

        public static double MultipleAsMatrix(Vector a, Vector b)
        {
            CheckVectorLength(a, b);

            var result = new double[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                result[i] = a[i] * b[i];
            }
            return result.Sum();
        }

        public static Vector Multiple(Vector a, Vector b)
        {
            CheckVectorLength(a, b);

            var result = new double[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                result[i] = a[i] * b[i];
            }
            return new Vector(result);
        }

        #endregion

        #region Matrix Operations

        public static Matrix Add(Matrix m, double b)
        {
            var result = new double[m.Rows, m.Columns];
            for (int i = 0; i < m.Rows; i++)
            {
                for (int j = 0; j < m.Columns; j++)
                {
                    result[i, j] = m[i, j] + b;
                }
            }
            return new Matrix(result);
        }

        public static Matrix Add(Matrix a, Matrix b)
        {
            CheckMatrixShape(a, b);

            var result = new double[a.Rows, a.Columns];
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Columns; j++)
                {
                    result[i, j] = a[i, j] + b[i, j];
                }
            }
            return new Matrix(result);
        }

        public static Matrix Multiple(Matrix m, double b)
        {
            var result = new double[m.Rows, m.Columns];
            for (int i = 0; i < m.Rows; i++)
            {
                for (int j = 0; j < m.Columns; j++)
                {
                    result[i, j] = m[i, j] * b;
                }
            }
            return new Matrix(result);
        }

        public static Matrix Multiple(Matrix a, Matrix b)
        {
            if (a.Columns != b.Rows)
                throw new Exception($"a.Columns={a.Columns} and b.Rows={b.Rows} are not equal!");

            var result = new double[a.Rows, b.Columns];
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < b.Columns; j++)
                {
                    result[i, j] = 0;
                    for (int k = 0; k < a.Columns; k++)
                    {
                        result[i, j] += (a[i, k] * b[k, j]);
                    }
                }
            }
            return new Matrix(result);
        }

        public static Matrix Multiple(Matrix a, Vector v)
        {
            var m = v.ToMatrix(true);
            return Multiple(a, m);
        }

        public static Matrix Multiple(Vector v, Matrix a)
        {
            var m = v.ToMatrix();
            return Multiple(m, a);
        }

        #endregion

        #region Helper Functions

        public static void CheckVectorLength(Vector a, Vector b)
        {
            if (a.Length != b.Length)
                throw new Exception($"vector a.Length={a.Length} and b.Length={b.Length} are not the equal!");
        }

        public static void CheckMatrixShape(Matrix a, Matrix b)
        {
            if(a.Rows != b.Rows || a.Columns != b.Columns)
                throw new Exception($"matrix shape a:[{a.Rows},{a.Columns}] and b:[{b.Rows},{b.Columns}] are not equal!");
        }

        #endregion
    }
}
