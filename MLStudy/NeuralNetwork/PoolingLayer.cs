using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace MLStudy
{
    public class PoolingLayer
    {
        public PoolingType PoolingType { get; private set; }
        public int Rows { get; private set; }
        public int Columns { get; private set; }
        public int Stride { get; private set; }
        private Func<Tensor, int, int, int, double> GetOut;

        public PoolingLayer(int rows = 2, int columns = 2, int stride = 2, PoolingType type = PoolingType.Max)
        {
            Rows = rows;
            Columns = columns;
            Stride = stride;
            PoolingType = type;

            if (type == PoolingType.Max)
                GetOut = GetMax;
            if (type == PoolingType.Mean)
                GetOut = GetMean;
        }

        public Tensor Forward(Tensor input)
        {
            var outRows = (input.Rows - Rows) / Stride + 1;
            var outColumns = (input.Columns - Columns) / Stride + 1;
            var result = new Tensor(outRows, outColumns, input.Depth);

            Parallel.For(0, result.Depth, i =>
            {
                for (int j = 0; j < result.Rows; j+=Stride)
                {
                    for (int k = 0; k < result.Columns; k+=Stride)
                    {
                        result[i, j, k] = GetOut(input, i, j * Stride, k * Stride);
                    }
                }
            });

            return result;
        }

        private double GetMax(Tensor input, int depthIndex, int startRow, int startColumns)
        {
            var result = double.MinValue;
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    if (input[depthIndex, startRow + i, startColumns + j] > result)
                        result = input[depthIndex, startRow + i, startColumns + j];
                }
            }
            return result;
        }

        private double GetMean(Tensor input, int depthIndex, int startRow, int startColumns)
        {
            var result = 0d;
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    result += input[depthIndex, startRow + i, startColumns + j];
                }
            }
            return result / (Rows * Columns);
        }
    }

    public enum PoolingType
    {
        Max,
        Mean
    }
}
