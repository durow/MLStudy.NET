using System;
using System.Collections.Generic;
using System.Text;

namespace MLSharp
{
    public class Matrix
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

        public double GetValue(int row, int column)
        {
            return values[row, column];
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

        public double Mean()
        {
            return Sum() / Length;
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
    }
}
