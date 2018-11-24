using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace MLStudy.Deep
{
    public sealed class MeanPooling : ILayer
    {
        public string Name { get; set; }
        public int Rows { get; private set; }
        public int Columns { get; private set; }
        public int RowStride { get; private set; }
        public int ColumnStride { get; private set; }
        public Tensor ForwardOutput { get; private set; }
        public Tensor BackwardOutput { get; private set; }

        private int samples;
        private int channels;
        private int outRows;
        private int outColumns;
        private int poolElements;

        public MeanPooling(int sizeAndStride)
            : this(sizeAndStride, sizeAndStride, sizeAndStride, sizeAndStride)
        { }

        public MeanPooling(int square, int stride)
            : this(square, square, stride, stride)
        { }

        public MeanPooling(int rows, int columns, int rowStride, int columnStride)
        {
            Rows = rows;
            Columns = columns;
            RowStride = rowStride;
            ColumnStride = columnStride;
        }

        public Tensor PrepareTrain(Tensor input)
        {
            PreparePredict(input);
            BackwardOutput = input.GetSameShape();
            return ForwardOutput;
        }

        public Tensor PreparePredict(Tensor input)
        {
            if (input.Rank != 4)
                throw new Exception("MaxPooling layer input rank must be 4!");

            outRows = (input.shape[2] - Rows) / RowStride + 1;
            outColumns = (input.shape[3] - Columns) / ColumnStride + 1;
            samples = input.shape[0];
            channels = input.shape[1];
            poolElements = outRows * outColumns;
            ForwardOutput = new Tensor(samples, channels, outRows, outColumns);

            return ForwardOutput;
        }

        public Tensor Forward(Tensor input)
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

        public Tensor Backward(Tensor error)
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

        public ILayer CreateSame()
        {
            return new MeanPooling(Rows, Columns, RowStride, ColumnStride);
        }

        private void PoolingChannel(Tensor input, int sample, int channel)
        {
            for (int row = 0; row < outRows; row++)
            {
                var inputRow = row * RowStride;
                for (int col = 0; col < outColumns; col++)
                {
                    var inputColumn = col * ColumnStride;
                    var sum = 0d;

                    for (int i = 0; i < Rows; i++)
                    {
                        for (int j = 0; j < Columns; j++)
                        {
                            sum += input[sample, channel, inputRow + i, inputColumn + j];
                        }
                    }

                    ForwardOutput[sample, channel, row, col] = sum / poolElements;
                }
            }
        }

        private void ErrorBP(Tensor error, int sample, int channel)
        {
            for (int row = 0; row < outRows; row++)
            {
                var inputRow = row * RowStride;
                for (int col = 0; col < outColumns; col++)
                {
                    var inputColumn = col * ColumnStride;
                    var e = error[sample, channel, row, col] / poolElements;
                    for (int i = 0; i < Rows; i++)
                    {
                        for (int j = 0; j < Columns; j++)
                        {
                            BackwardOutput[sample, channel, inputRow + i, inputColumn + j] += e;
                        }
                    }
                }
            }
        }
    }
}
