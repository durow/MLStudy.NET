﻿using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace MLStudy.Deep
{
    public sealed class MaxPooling : PoolingLayer
    {
        private TensorOld maxPositions;

        public MaxPooling(int sizeAndStride)
            : this(sizeAndStride, sizeAndStride, sizeAndStride, sizeAndStride)
        { }

        public MaxPooling(int square, int stride)
            : this(square, square, stride, stride)
        { }

        public MaxPooling(int rows, int columns, int rowStride, int columnStride)
        {
            Rows = rows;
            Columns = columns;
            RowStride = rowStride;
            ColumnStride = columnStride;
        }

        public override TensorOld PrepareTrain(TensorOld input)
        {
            PreparePredict(input);

            BackwardOutput = input.GetSameShape();
            maxPositions = new TensorOld(samples, channels, outRows * outColumns, 2);

            return ForwardOutput;
        }

        public override TensorOld PreparePredict(TensorOld input)
        {
            if (input.Rank != 4)
                throw new Exception("MaxPooling layer input rank must be 4!");

            outRows = (input.shape[2] - Rows) / RowStride + 1;
            outColumns = (input.shape[3] - Columns) / ColumnStride + 1;
            samples = input.shape[0];
            channels = input.shape[1];
            ForwardOutput = new TensorOld(samples, channels, outRows, outColumns);

            return ForwardOutput;
        }

        public override TensorOld Forward(TensorOld input)
        {
            Parallel.For(0, ForwardOutput.shape[0], sampleIndex =>
            {
                Parallel.For(0, ForwardOutput.shape[1], channel =>
                {
                    PoolingChannel(input, sampleIndex, channel);
                });
            });
            return ForwardOutput;
        }

        public override TensorOld Backward(TensorOld error)
        {
            BackwardOutput.Clear();

            Parallel.For(0, ForwardOutput.shape[0], sampleIndex =>
            {
                Parallel.For(0, ForwardOutput.shape[1], channel =>
                {
                    ErrorBP(error, sampleIndex, channel);
                });
            });
            return BackwardOutput;
        }

        public override ILayer CreateSame()
        {
            return new MaxPooling(Rows, Columns, RowStride, ColumnStride);
        }

        private void PoolingChannel(TensorOld input, int sample, int channel)
        {
            var maxIndex = 0;
            for (int row = 0; row < outRows; row++)
            {
                var inputRow = row * RowStride;
                for (int col = 0; col < outColumns; col++)
                {
                    var inputColumn = col * ColumnStride;
                    var max = input[sample, channel, inputRow, inputColumn];
                    var maxRow = inputRow;
                    var maxColumn = inputColumn;

                    for (int i = 0; i < Rows; i++)
                    {
                        for (int j = 0; j < Columns; j++)
                        {
                            if (input[sample, channel, inputRow + i, inputColumn + j] > max)
                            {
                                max = input[sample, channel, inputRow + i, inputColumn + j];
                                maxRow = inputRow + i;
                                maxColumn = inputColumn + j;
                            }
                        }
                    }

                    ForwardOutput[sample, channel, row, col] = max;
                    if (maxPositions != null)
                    {
                        maxPositions[sample, channel, maxIndex, 0] = maxRow;
                        maxPositions[sample, channel, maxIndex, 1] = maxColumn;
                        maxIndex++;
                    }
                }
            }
        }

        private void ErrorBP(TensorOld error, int sample, int channel)
        {
            var maxIndex = 0;
            for (int i = 0; i < outRows; i++)
            {
                for (int j = 0; j < outColumns; j++)
                {
                    var e = error[sample, channel, i, j];
                    var maxRow = maxPositions[sample, channel, maxIndex, 0];
                    var maxCol = maxPositions[sample, channel, maxIndex, 1];
                    BackwardOutput[sample, channel, (int)maxRow, (int)maxCol] += e;

                    maxIndex++;
                }
            }
        }
    }
}
