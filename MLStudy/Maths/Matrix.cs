using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public struct Matrix
    {
        private double[,] values;

        public int Length { get { return values.Length; } }
        public int Rows { get { return values.GetLength(0); } }
        public int Columns { get { return values.GetLength(1); } }

        public double this[int row, int column]
        {
            get
            {
                return GetValue(row, column);
            }
            set
            {
                values[row, column] = value;
            }
        }

        public Vector this[int index]
        {
            get
            {
                return GetRow(index);
            }
        }

        public Matrix(int rows, int columns)
        {
            values = new double[rows, columns];
        }

        public Matrix(int rows, int columns, double defaultValue)
        {
            values = new double[rows, columns];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    values[i, j] = defaultValue;
                }
            }
        }

        public Matrix(double[,] values)
        {
            this.values = values;
        }

        public Vector GetRow(int index)
        {
            var columns = values.GetLength(1);
            var result = new double[columns];
            for (int i = 0; i < columns; i++)
            {
                result[i] = values[index, i];
            }

            return new Vector(result);
        }

        public Vector GetColumn(int index)
        {
            var rows = values.GetLength(0);
            var result = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                result[i] = values[i, index];
            }

            return new Vector(result);
        }

        public Matrix GetRange(int startRow, int rowLength, int startColumn, int columnLength)
        {
            var result = new Matrix(rowLength, columnLength);

            for (int i = 0; i < rowLength; i++)
            {
                for (int j = 0; j < columnLength; j++)
                {
                    result[i, j] = values[startRow + i, startColumn + j];
                }
            }

            return result;
        }

        public double GetValue(int row, int column)
        {
            return values[row, column];
        }

        public Matrix GetSameShape()
        {
            return new Matrix(Rows, Columns);
        }

        public override bool Equals(object obj)
        {
            if (!(obj is Matrix m))
                return false;

            if (Rows != m.Rows)
                return false;

            if (Columns != m.Columns)
                return false;

            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    if (values[i, j] != m[i, j])
                        return false;
                }
            }

            return true;
        }

        public override string ToString()
        {
            var rows = new List<string>();
            for (int i = 0; i < Rows; i++)
            {
                rows.Add(this[i].ToString());
            }
            var content = string.Join(",\n", rows);
            return $"[{content}]";
        }

        public double Sum()
        {
            var result = 0d;
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    result += values[i, j];
                }
            }
            return result;
        }

        public Vector SumRow()
        {
            var result = new Vector(Columns);
            for (int i = 0; i < Rows; i++)
            {
                result += GetRow(i);
            }
            return result;
        }

        public Vector SumColumn()
        {
            var result = new Vector(Rows);
            for (int i = 0; i < Columns; i++)
            {
                result += GetColumn(i);
            }
            return result;
        }

        public double Mean()
        {
            return Sum() / Length;
        }

        public Vector MeanRow()
        {
            return SumRow() / Rows;
        }

        public Vector MeanColumn()
        {
            return SumColumn() / Columns;
        }

        public Matrix Transpose(bool changeLocal = false)
        {
            var result = new double[Columns, Rows];
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    result[j, i] = values[i, j];
                }
            }

            if (changeLocal)
                values = result;

            return new Matrix(result);
        }

        public Vector ToVector()
        {
            if (Rows == 1)
                return GetRow(0);
            if (Columns == 1)
                return GetColumn(0);

            throw new Exception($"This matrix [{Rows},{Columns}] can't convert to Vector!");
        }

        public Matrix ApplyFunction(Func<double,double> func)
        {
            return MatrixOperations.Instance.Apply(this, func);
        }

        #region Operations

        public static Matrix operator +(Matrix m, double a)
        {
            return MatrixOperations.Instance.Add(m, a);
        }

        public static Matrix operator +(double a, Matrix m)
        {
            return MatrixOperations.Instance.Add(m, a);
        }

        public static Matrix operator +(Matrix m, Vector a)
        {
            return MatrixOperations.Instance.Add(m, a);
        }

        public static Matrix operator +(Vector a, Matrix m)
        {
            return MatrixOperations.Instance.Add(a, m);
        }

        public static Matrix operator +(Matrix a, Matrix b)
        {
            return MatrixOperations.Instance.Add(a, b);
        }

        public static Matrix operator -(Matrix a, Matrix b)
        {
            return MatrixOperations.Instance.Minus(a, b);
        }

        public static Matrix operator -(Matrix m, double a)
        {
            return MatrixOperations.Instance.Minus(m, a);
        }

        public static Matrix operator *(Matrix m, double b)
        {
            return MatrixOperations.Instance.Multiple(m, b);
        }

        public static Matrix operator *(double b, Matrix m)
        {
            return MatrixOperations.Instance.Multiple(m, b);
        }

        public static Matrix operator *(Matrix m, Vector v)
        {
            return MatrixOperations.Instance.Multiple(m, v);
        }

        public static Matrix operator *(Vector v, Matrix m)
        {
            return MatrixOperations.Instance.Multiple(v, m);
        }

        public static Matrix operator *(Matrix a, Matrix b)
        {
            return MatrixOperations.Instance.Multiple(a, b);
        }

        public static Matrix operator /(Matrix m, double a)
        {
            return MatrixOperations.Instance.Divide(m, a);
        }

        #endregion
    }
}
