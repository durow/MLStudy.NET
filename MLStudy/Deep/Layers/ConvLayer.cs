using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace MLStudy.Deep
{
    public sealed class ConvLayer : ILayer, IOptimizable
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
        public TensorOld Filters { get; private set; }
        public TensorOld Bias { get; private set; }
        public TensorOld ForwardOutput { get; private set; }
        public TensorOld BackwardOutput { get; private set; }
        public TensorOld FiltersGradient { get; private set; }
        public TensorOld BiasGradient { get; private set; }
        public TensorOld PaddingInput { get; private set; }

        private List<ConvLayer> mirrorList = new List<ConvLayer>();
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

            mirrorList.Add(this);
        }

        public TensorOld PrepareTrain(TensorOld input)
        {
            PreparePredict(input);
            BackwardOutput = input.GetSameShape();
            FiltersGradient = Filters.GetSameShape();
            BiasGradient = Bias.GetSameShape();
            return ForwardOutput;
        }

        public TensorOld PreparePredict(TensorOld input)
        {
            if (input.Rank != 4)
                throw new Exception("Convolutional layer input rank must be 4!");

            samples = input.shape[0];
            channels = input.shape[1];
            inputRows = input.shape[2];
            inputColumns = input.shape[3];
            outRows = (inputRows + 2 * RowPadding - FilterRows) / RowStride + 1;
            outColumns = (inputColumns + 2 * ColumnPadding - FilterColumns) / ColumnStride + 1;
            ForwardOutput = new TensorOld(samples, FilterCount, outRows, outColumns);
            if (Filters == null)
                SetFilters(TensorOld.RandGaussian(FilterCount, channels, FilterRows, FilterColumns));
            if (Bias == null)
                SetBias(TensorOld.Zeros(FilterCount));
            PaddingInput = TensorOld.Values(PaddingValue, samples, channels, inputRows + 2 * RowPadding, inputColumns + 2 * ColumnPadding);
            return ForwardOutput;
        }

        public TensorOld Forward(TensorOld input)
        {
            SetPaddingInput(input);

            Parallel.For(0, samples, sampleIndex =>
            {
                Parallel.For(0, FilterCount, filterIndex =>
                {
                    Forward(input, sampleIndex, filterIndex);
                });
            });
            return ForwardOutput;
        }

        public TensorOld Backward(TensorOld error)
        {
            ErrorBackward(error);
            ComputeGradient(error);

            return BackwardOutput;
        }

        public void Optimize(IOptimizer optimizer)
        {
            optimizer.Optimize(Filters, FiltersGradient);
            optimizer.Optimize(Bias, BiasGradient);
        }

        public void Regularize(IRegularizer regularizer)
        {
            regularizer.Regularize(Filters, FiltersGradient);
        }

        public ILayer CreateMirror()
        {
            var result = (ConvLayer)CreateSame();
            result.Filters = Filters;
            result.Bias = Bias;
            result.mirrorList = mirrorList;
            mirrorList.Add(result);
            return result;
        }

        public ILayer CreateSame()
        {
            return new ConvLayer(FilterCount, FilterRows, FilterColumns, RowStride, ColumnStride, RowPadding, ColumnPadding);
        }

        public void SetFilters(TensorOld filters)
        {
            if (Filters == null)
            {
                foreach (var item in mirrorList)
                {
                    item.Filters = filters;
                }
            }
            TensorOld.CheckShape(filters, Filters);
            Array.Copy(filters.values, 0, Filters.values, 0, Filters.ElementCount);
        }

        public void SetBias(TensorOld bias)
        {
            if (Bias == null)
            {
                foreach (var item in mirrorList)
                {
                    item.Bias = bias;
                }
            }
            TensorOld.CheckShape(Bias, bias);
            Array.Copy(bias.GetRawValues(), 0, Bias.GetRawValues(), 0, Bias.ElementCount);
        }

        private void Forward(TensorOld input, int sampleIndex, int filterIndex)
        {
            Parallel.For(0, outRows, row =>
            {
                var startRow = row * RowStride;
                Parallel.For(0, outColumns, col =>
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
                                sum += PaddingInput[sampleIndex, k, inputRow, inputCol] * Filters[filterIndex, k, i, j];
                            }
                        }
                    }
                    ForwardOutput[sampleIndex, filterIndex, row, col] = sum + Bias.GetRawValues()[filterIndex];
                });
            });
        }

        private void ErrorBackward(TensorOld error)
        {
            BackwardOutput.Clear();

            Parallel.For(0, samples, sampleIndex =>
            {
                Parallel.For(0, FilterCount, filterIndex =>
                {
                    ErrorBackward(error, sampleIndex, filterIndex);
                });
            });
        }

        private void ErrorBackward(TensorOld error, int sampleIndex, int filterIndex)
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
                                BackwardOutput[sampleIndex, k, inputRow, inputCol] += Filters[filterIndex, k, i, j] * e;
                            }
                        }
                    }
                }
            }
        }

        private void ComputeGradient(TensorOld error)
        {
            FiltersGradient.Clear();

            Parallel.For(0, samples, sampleIndex =>
            {
                Parallel.For(0, FilterCount, filterIndex =>
                {
                    ComputeGradient(error, sampleIndex, filterIndex);
                });
            });
        }

        private void ComputeGradient(TensorOld error, int sampleIndex, int filterIndex)
        {
            var bias = 0d;
            for (int row = 0; row < outRows; row++)
            {
                var startRow = row * RowStride;
                for (int col = 0; col < outColumns; col++)
                {
                    var startCol = col * ColumnStride;
                    var e = error[sampleIndex, filterIndex, row, col];
                    bias += e;

                    for (int i = 0; i < FilterRows; i++)
                    {
                        var inputRow = startRow + i;
                        for (int j = 0; j < FilterColumns; j++)
                        {
                            var inputCol = startCol + j;

                            for (int k = 0; k < channels; k++)
                            {
                                FiltersGradient[filterIndex, k, i, j] += PaddingInput[sampleIndex, k, inputRow, inputCol] * e;
                            }
                        }
                    }
                }
            }
            BiasGradient[filterIndex] = bias;
        }

        private double GetInputValue(TensorOld input, int sample, int channel, int row, int col)
        {
            if (row < RowPadding || col < ColumnPadding)
                return PaddingValue;

            row -= RowPadding;
            col -= ColumnPadding;

            if (row >= inputRows || col >= inputColumns)
                return PaddingValue;

            return input[sample, channel, row, col];
        }

        private void SetPaddingInput(TensorOld input)
        {
            var inputData = input.GetRawValues();
            var paddingData = PaddingInput.GetRawValues();

            for (int sample = 0; sample < samples; sample++)
            {
                for (int channel = 0; channel < channels; channel++)
                {
                    for (int i = 0; i < input.shape[2]; i++)
                    {
                        var inputStart = input.GetRawOffset(sample, channel, i, 0);
                        var paddingStart = PaddingInput.GetRawOffset(sample, channel, i + RowPadding, ColumnPadding);
                        Array.Copy(inputData, inputStart, paddingData, paddingStart, input.shape[3]);
                    }
                }
            }
        }
    }
}
