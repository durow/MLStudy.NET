using MLStudy.Activations;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace MLStudy
{
    public class ConvolutionalLayer
    {
        public int InputDepth { get; private set; }
        public int FilterRows { get; private set; }
        public int FilterColumns { get; private set; }
        public int FilterCount { get; private set; }
        public List<Filter> Filters { get; private set; } = new List<Filter>();
        public Tensor ForwardInput { get; private set; }
        public Tensor ForwardOutput { get; private set; }
        public Activation Activation { get; set; } = new ReLU();

        public int Stride { get; set; } = 1;
        public int Padding { get; set; } = 0;

        public ConvolutionalLayer(int inputDepth, int filterRows, int filterColumns, int filterCount)
        {
            InputDepth = inputDepth;
            FilterRows = filterRows;
            FilterColumns = filterColumns;
            FilterCount = filterCount;

            for (int i = 0; i < FilterCount; i++)
            {
                Filters.Add(new Filter(filterRows, filterColumns, inputDepth));
            }
        }

        public Tensor Forward(Tensor input)
        {
            ForwardInput = DoPadding(input);
            var (outRows, outColumns) = GetOutputSize();
            var output = new Tensor(outRows, outColumns, FilterCount);

            Parallel.For(0, FilterCount, i =>
            {
                for (int j = 0; j < input.Rows; j+=Stride)
                {
                    for (int k = 0; k < input.Columns; k+=Stride)
                    {
                        output[i, j, k] = TensorOperations.Instance.Convolute(ForwardInput, Filters[i].Weights, j, k) + Filters[i].Bias;
                    }
                }
            });

            ForwardOutput = Activation.Forward(output);
            return ForwardOutput;
        }

        private (int, int) GetOutputSize()
        {
            var outRows = (ForwardInput.Rows - FilterRows) / Stride + 1;
            var outColumns = (ForwardInput.Columns - FilterColumns) / Stride + 1;
            return (outRows, outColumns);
        }

        private Tensor DoPadding(Tensor input)
        {
            if (Padding == 0)
                return input;

            var result = new Tensor(input.Rows + Padding * 2, input.Columns + Padding * 2, input.Depth);
            for (int i = 0; i < input.Depth; i++)
            {
                for (int j = 0; j < input.Rows; j++)
                {
                    for (int k = 0; k < input.Columns; k++)
                    {
                        result[i, j + Padding, k + Padding] = input[i, j, k];
                    }
                }
            }
            return result;
        }
    }

    public class Filter
    {
        public Tensor Weights { get; private set; }
        public int Rows { get { return Weights.Rows; } }
        public int Columns { get { return Weights.Columns; } }
        public int Depth { get { return Weights.Depth; } }
        public double Bias { get; private set; }

        public Filter(int rows, int columns, int depth)
        {
            Weights = new Tensor(depth, rows, columns);
        }

        public void SetWeights(Tensor weights)
        {
            Weights = weights;
        }

        public void SetBias(double bias)
        {
            Bias = bias;
        }
    }
}
