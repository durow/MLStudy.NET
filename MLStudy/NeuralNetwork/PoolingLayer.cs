/*
 */

using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace MLStudy
{
    public class PoolingLayer
    {
        public PoolingType PoolingType { get; private set; }
        public int Rows { get; private set; }
        public int Columns { get; private set; }
        public int Stride { get; private set; }
        public int InputRows { get; private set; }
        public int InputColumns { get; private set; }
        public int InputDepth { get; private set; }
        public int OutputRows { get; private set; }
        public int OutputColumns { get; private set; }

        private Func<Tensor, int, int, int, double> GetOut;
        private List<List<(int,int)>> maxPosition;

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
            InputRows = input.Rows;
            InputColumns = input.Columns;
            InputDepth = input.Depth;

            OutputRows = (input.Rows - Rows) / Stride + 1;
            OutputColumns = (input.Columns - Columns) / Stride + 1;

            InitMaxPositionList();

            var result = new Tensor(OutputRows, OutputColumns, input.Depth);
            Parallel.For(0, result.Depth, i =>
            {
                for (int j = 0; j < result.Rows; j++)
                {
                    for (int k = 0; k < result.Columns; k++)
                    {
                        result[i, j, k] = GetOut(input, i, j * Stride, k * Stride);
                    }
                }
            });
            return result;
        }

        private void InitMaxPositionList()
        {
            if (PoolingType == PoolingType.Max)
            {
                maxPosition = new List<List<(int, int)>>(InputDepth);
                for (int i = 0; i < InputDepth; i++)
                {
                    maxPosition[i] = new List<(int, int)>(OutputRows * OutputColumns);
                }
            }
            else
                maxPosition = null;
        }

        public Tensor Backward(Tensor outputError)
        {
            if (PoolingType == PoolingType.Max)
                return BackwardMax(outputError);
            else
                return BackwardMean(outputError);
        }

        private Tensor BackwardMean(Tensor outputError)
        {
            var result = new Tensor(InputRows, InputColumns, InputDepth);
            Parallel.For(0, result.Depth, i =>
            {
                for (int j = 0; j < outputError.Rows; j++)
                {
                    for (int k = 0; k < outputError.Columns; k++)
                    {
                        var startRow = j * Stride;
                        var startColumn = k * Stride;
                        var error = outputError[i, j, k];
                        SetMeanError(result, startRow, startColumn, i, error);
                    }
                }
            });
            return result;
        }

        private Tensor BackwardMax(Tensor outputError)
        {
            var result = new Tensor(InputRows, InputColumns, InputDepth);
            Parallel.For(0, result.Depth, i =>
            {
                var position = maxPosition[i];
                for (int j = 0; j < outputError.Rows; j++)
                {
                    for (int k = 0; k < outputError.Columns; k++)
                    {
                        var index = j * outputError.Rows + k;
                        var row = position[index].Item1;
                        var column = position[index].Item2;
                        result[i, row, column] += outputError[i, j, k];
                    }
                }
            });
            return result;
        }

        private void SetMeanError(Tensor inputError, int startRow, int startColumn, int depth, double error)
        {
            var meanError = error / (Rows * Columns);
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    inputError[depth, startRow + i, startRow + j] += meanError;
                }
            }
        }

        private void SetMeanMax(Tensor inputError, int startRow, int startColumn, int depth, double error)
        {
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                }
            }
        }

        private double GetMax(Tensor input, int depthIndex, int startRow, int startColumns)
        {
            var result = double.MinValue;
            var maxRow = 0;
            var maxColumn = 0;

            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    var row = startRow + i;
                    var column = startColumns + j;
                    if (input[depthIndex, row, column] > result)
                    {
                        result = input[depthIndex, row, column];
                        maxRow = row;
                        maxColumn = column;
                    }
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
