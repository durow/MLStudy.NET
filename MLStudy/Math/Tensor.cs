﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;


namespace MLStudy
{
    public class Tensor
    {

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

        public static Vector Minus(Vector v, double b)
        {
            var result = new double[v.Length];
            for (int i = 0; i < v.Length; i++)
            {
                result[i] = v[i] - b;
            }
            return new Vector(result);
        }

        public static Vector Minus(Vector a, Vector b)
        {
            CheckVectorLength(a, b);

            var result = new double[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                result[i] = a[i] - b[i];
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

        public static Vector Divide(Vector v, double b)
        {
            var result = new double[v.Length];
            for (int i = 0; i < v.Length; i++)
            {
                result[i] = v[i] / b;
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

        public static Matrix Add(Matrix a, Vector b)
        {
            if (a.Rows != b.Length)
                throw new Exception("not the same size!");

            var result = new Matrix(a.Rows, a.Columns);
            for (int i = 0; i < a.Columns; i++)
            {
                for (int j = 0; j < a.Rows; j++)
                {
                    result[j, i] = a[j, i] + b[j];
                }
            }

            return result;
        }

        public static Matrix Add(Vector a, Matrix b)
        {
            if (b.Rows != a.Length)
                throw new Exception("not the same size!");

            var result = new Matrix(b.Rows, b.Columns);
            for (int i = 0; i < b.Rows; i++)
            {
                for (int j = 0; j < b.Columns; j++)
                {
                    result[i, j] = a[j] + b[i, j];
                }
            }

            return result;
        }

        public static Matrix Minus(Matrix m, double b)
        {
            var result = new double[m.Rows, m.Columns];
            for (int i = 0; i < m.Rows; i++)
            {
                for (int j = 0; j < m.Columns; j++)
                {
                    result[i, j] = m[i, j] - b;
                }
            }
            return new Matrix(result);
        }

        public static Matrix Minus(Matrix a, Matrix b)
        {
            CheckMatrixShape(a, b);

            var result = new double[a.Rows, a.Columns];
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Columns; j++)
                {
                    result[i, j] = a[i, j] - b[i, j];
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

        public static Matrix MultipleElementWise(Matrix a, Matrix b)
        {
            CheckMatrixShape(a, b);

            var result = a.GetSameShape();
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Columns; j++)
                {
                    result[i, j] = a[i, j] * b[i, j];
                }
            }
            return result;
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

        public static Matrix Divide(Matrix m, double b)
        {
            var result = new double[m.Rows, m.Columns];
            for (int i = 0; i < m.Rows; i++)
            {
                for (int j = 0; j < m.Columns; j++)
                {
                    result[i, j] = m[i, j] / b;
                }
            }
            return new Matrix(result);
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