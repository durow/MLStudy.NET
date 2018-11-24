using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace MLStudy.Deep
{
    public sealed class ConvLayer : ILayer
    {
        public string Name { get; set; }
        public int FilterCount { get; private set; }
        public int FilterRows { get; private set; }
        public int FilterColumns { get; private set; }
        public int RowStride { get; private set; }
        public int ColumnStride { get; private set; }
        public int RowPadding { get; private set; }
        public int ColumnPadding { get; private set; }
        public double PaddingValue { get; set; } = 0;
        public Tensor Filters { get; private set; }
        public Tensor Bias { get; private set; }
        public Tensor ForwardOutput { get; private set; }
        public Tensor BackwardOutput { get; private set; }
        public Tensor FiltersGradient { get; private set; }
        public Tensor BiasGradient { get; private set; }
        public Tensor Input { get; private set; }

        private int samples;
        private int channels;
        private int inputRows;
        private int inputColumns;
        private int outRows;
        private int outColumns;

        public ConvLayer(int filterCount, int square, int stride, int padding = 0)
            : this(filterCount, square, square, stride, stride, padding, padding)
        { }

        public ConvLayer(int filterCount, int filterRows, int filterColumns, int rowStride, int columnStride, int rowPadding = 0, int columnPadding = 0)
        {
            FilterCount = filterCount;
            FilterRows = filterRows;
            FilterColumns = filterColumns;
            RowStride = rowStride;
            ColumnStride = columnStride;
            RowPadding = rowPadding;
            ColumnPadding = columnPadding;
        }

        public Tensor PrepareTrain(Tensor input)
        {
            BackwardOutput = input.GetSameShape();
            FiltersGradient = Filters.GetSameShape();
            BiasGradient = Bias.GetSameShape();
            Input = input;
            return PreparePredict(input);
        }

        public Tensor PreparePredict(Tensor input)
        {
            if (input.Rank != 4)
                throw new Exception("Convolutional layer input rank must be 4!");

            samples = input.shape[0];
            channels = input.shape[1];
            inputRows = input.shape[2];
            inputColumns = input.shape[3];
            outRows = (inputRows + 2 * RowPadding - FilterRows) / RowStride + 1;
            outColumns = (inputColumns + 2 * ColumnPadding - FilterColumns) / ColumnStride + 1;
            ForwardOutput = new Tensor(samples, FilterCount, outRows, outColumns);
            Filters = Tensor.RandGaussian(FilterCount, channels, FilterRows, FilterColumns);
            Bias = Tensor.Zeros(FilterCount);
            return ForwardOutput;
        }

        public Tensor Forward(Tensor input)
        {
            Parallel.For(0, samples, sampleIndex =>
            {
                Parallel.For(0, FilterCount, filterIndex =>
                {
                    Forward(input, sampleIndex, filterIndex);
                });
            });
            return ForwardOutput;
        }

        public Tensor Backward(Tensor error)
        {
            BackwardOutput.Clear();

            ErrorBackward(error);
            ComputeGradient(error);

            return BackwardOutput;
        }

        public ILayer CreateSame()
        {
            return new ConvLayer(FilterCount, FilterRows, FilterColumns, RowStride, ColumnStride, RowPadding, ColumnPadding);
        }

        private void Forward(Tensor input, int sampleIndex, int filterIndex)
        {
            for (int row = 0; row < outRows; row++)
            {
                var startRow = row * RowStride;
                for (int col = 0; col < outColumns; col++)
                {
                    var startCol = col * ColumnStride;
                    var sum = 0d;

                    for (int i = 0; i < FilterRows; i++)
                    {
                        var inputRow = startRow + i;
                        for (int j = 0; j < FilterColumns; j++)
                        {
                            var inputCol = startCol + j;

                            for (int k = 0; k < channels; k++)
                            {
                                sum += GetInputValue(input, sampleIndex, k, inputRow, inputCol) * Filters[filterIndex, k, i, j];
                            }
                        }
                    }
                    ForwardOutput[sampleIndex, filterIndex, row, col] = sum;
                }
            }
        }

        private void ErrorBackward(Tensor error)
        {
            Parallel.For(0, samples, sampleIndex =>
            {
                Parallel.For(0, FilterCount, filterIndex =>
                {
                    ErrorBackward(error, sampleIndex, filterIndex);
                });
            });
        }

        private void ErrorBackward(Tensor error, int sampleIndex, int filterIndex)
        {
            for (int row = 0; row < outRows; row++)
            {
                var startRow = row * RowStride;
                for (int col = 0; col < outColumns; col++)
                {
                    var startCol = col * ColumnStride;
                    var e = error[sampleIndex, filterIndex, row, col];

                    for (int i = 0; i < FilterRows; i++)
                    {
                        var inputRow = startRow + i;
                        for (int j = 0; j < FilterColumns; j++)
                        {
                            var inputCol = startCol + j;

                            if (inputRow < RowPadding || inputCol < ColumnPadding)
                                continue;

                            inputRow -= RowPadding;
                            inputCol -= ColumnPadding;
                            if (inputRow >= inputRows || inputCol >= inputColumns)
                                continue;

                            for (int k = 0; k < channels; k++)
                            {
                                BackwardOutput[sampleIndex, k, inputRow, inputCol] += Input[sampleIndex, k, inputRow, inputCol] * e;
                            }
                        }
                    }
                }
            }
        }

        private void ComputeGradient(Tensor error)
        {
            Parallel.For(0, samples, sampleIndex =>
            {
                Parallel.For(0, FilterCount, filterIndex =>
                {
                    ComputeGradient(error, sampleIndex, filterIndex);
                });
            });
        }

        private void ComputeGradient(Tensor error, int sampleIndex, int filterIndex)
        {

        }

        private double GetInputValue(Tensor input, int sample, int channel, int row, int col)
        {
            if (row < RowPadding || col < ColumnPadding)
                return PaddingValue;

            row -= RowPadding;
            col -= ColumnPadding;

            if (row >= inputRows || col >= inputColumns)
                return PaddingValue;

            return input[sample, channel, row, col];
        }
    }
}
