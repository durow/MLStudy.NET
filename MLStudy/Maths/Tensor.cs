using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;


namespace MLStudy
{
    public struct Tensor
    {
        private double[,,] values;
        public int Dimension => 3;
        public int Rows { get { return values.GetLength(1); } }
        public int Columns { get { return values.GetLength(2); } }
        public int Depth { get { return values.GetLength(0); } }
        public Matrix this[int depthIndex]
        {
            get
            {
                return GetMatrix(depthIndex);
            }
        }
        public double this[int depthIndex, int rowIndex, int columnIndex]
        {
            get
            {
                return values[depthIndex, rowIndex, columnIndex];
            }
            set
            {
                values[depthIndex, rowIndex, columnIndex] = value;
            }
        }

        public Tensor(int rows, int columns, int depth)
        {
            values = new double[depth, rows, columns];
        }
        
        public Tensor(params Matrix[] matrixes)
        {
            if (matrixes.Length == 0)
            {
                values = new double[0, 0, 0];
                return;
            }

            var rows = matrixes[0].Rows;
            var columns = matrixes[0].Columns;
            CheckMatrixShape(matrixes, rows, columns);

            values = new double[matrixes.Length, matrixes[0].Rows, matrixes[1].Columns];
            for (int i = 0; i < matrixes.Length; i++)
            {
                for (int j = 0; j < rows; j++)
                {
                    for (int k = 0; k < columns; k++)
                    {
                        values[i, j, k] = matrixes[i][j, k];
                    }
                }
            }
        }

        public Matrix GetMatrix(int depthIndex)
        {
            var result = new Matrix(Rows, Columns);
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    result[i, j] = values[depthIndex, i, j];
                }
            }
            return result;
        }

        public Tensor GetSameShape()
        {
            return new Tensor(Depth, Rows, Columns);
        }

        public Tensor ApplyFunction(Func<double,double> function)
        {
            return TensorOperations.Instance.Apply(this, function);
        }

        private static void CheckMatrixShape(Matrix[] matrixes, int rows, int columns)
        {
            for (int i = 0; i < matrixes.Length; i++)
            {
                if (matrixes[i].Rows != rows || matrixes[i].Columns != columns)
                    throw new Exception("Matrixes are not the same shape!");
            }
        }
    }
}
