/*
 * Description:Convolutional layer
 * Author:YunXiao An
 * Date:2018.11.02
 */

using MLStudy.Abstraction;
using MLStudy.Activations;
using MLStudy.Optimization;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace MLStudy
{
    public class ConvolutionalLayer: CNNLayer
    {
        public double LearningRate
        {
            get
            {
                return Optimizer.LearningRate;
            }
            set
            {
                Optimizer.LearningRate = value;
            }
        }
        public int InputDepth { get; private set; }
        public int FilterRows { get; private set; }
        public int FilterColumns { get; private set; }
        public int FilterCount { get; private set; }
        public int Stride { get; set; } = 1;
        public int Padding { get; set; } = 0;

        public List<Filter> Filters { get; private set; } = new List<Filter>();
        public Tensor3 ForwardInputPadding { get; private set; }
        public Tensor3 ForwardOutput { get; private set; }
        public Tensor3 LinearError { get; private set; }
        public Activation Activation { get; set; } = new ReLU();
        public GradientOptimizer Optimizer { get; private set; } = new NormalDescent();

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

        public override Tensor3 Forward(Tensor3 input)
        {
            ForwardInputPadding = DoPadding(input);
            var (outRows, outColumns) = GetOutputSize();
            var output = new Tensor3(outRows, outColumns, FilterCount);

            Parallel.For(0, FilterCount, i =>
            {
                for (int j = 0; j < output.Rows; j++)
                {
                    for (int k = 0; k < output.Columns; k++)
                    {
                        output[i, j, k] = TensorOperations.Instance.Convolute(ForwardInputPadding, Filters[i].Weights, j * Stride, k * Stride) + Filters[i].Bias;
                    }
                }
            });

            ForwardOutput = Activation.Forward(output);
            return ForwardOutput;
        }

        public override Tensor3 Backward(Tensor3 outputError)
        {
            var inputError = ErrorBP(outputError);
            UpdateWeightsBias();
            return inputError;
        }

        private Tensor3 ErrorBP(Tensor3 outputError)
        {
            LinearError = Activation.Backward(ForwardOutput, outputError);
            var inputError = ForwardInputPadding.GetSameShape();

            var errorList = new Tensor3[FilterCount];
            Parallel.For(0, FilterCount, i =>
            {
                errorList[i] = GetInputErrorByFilter(outputError, i);
            });

            var result = ForwardInputPadding.GetSameShape();
            for (int i = 0; i < errorList.Length; i++)
            {
                result += errorList[i];
            }
            return UnPadding(result);
        }

        private Tensor3 GetInputErrorByFilter(Tensor3 outputError, int filterIndex)
        {
            var result = ForwardInputPadding.GetSameShape();
            var filter = Filters[filterIndex];
            for (int i = 0; i < outputError.Rows; i++)
            {
                for (int j = 0; j < outputError.Columns; j++)
                {
                    var error = filter.Weights * outputError[filterIndex, i, j];
                    var startRow = i * Stride;
                    var startColumn = j * Stride;
                    TensorOperations.Instance.AddLocal(result, error, startRow, startColumn);
                }
            }
            return result;
        }

        private Tensor3 UnPadding(Tensor3 paddingTensor)
        {
            var result = new Tensor3(paddingTensor.Rows - Padding * 2, paddingTensor.Columns - Padding * 2, paddingTensor.Depth);
            for (int i = 0; i < result.Depth; i++)
            {
                for (int j = 0; j < result.Rows; j++)
                {
                    for (int k = 0; k < result.Columns; k++)
                    {
                        result[i, j, k] = paddingTensor[i, Padding + j, Padding + k];
                    }
                }
            }
            return result;
        }

        private void UpdateWeightsBias()
        {
            Parallel.For(0, FilterCount, i =>
            {
                var filter = Filters[i];
                var gradient = GetFilterGradient(i);
                var biasGradient = gradient.Mean();
                var weights = Optimizer.GradientDescent(filter.Weights, gradient);
                var bias = Optimizer.GradientDescent(filter.Bias, biasGradient);
                filter.SetWeights(weights);
                filter.SetBias(bias);
            });
        }

        private Tensor3 GetFilterGradient(int filterIndex)
        {
            var filter = Filters[filterIndex];
            var result = filter.Weights.GetSameShape();

            for (int i = 0; i < LinearError.Rows; i++)
            {
                for (int j = 0; j < LinearError.Columns; j++)
                {
                    var error = TensorOperations.Instance.PartMultiple(ForwardInputPadding, LinearError[filterIndex,i,j], i * Stride, j * Stride, filter.Rows, filter.Columns);
                    result += error;
                }
            }
            return result / (LinearError.Rows * LinearError.Columns);
        }

        private (int, int) GetOutputSize()
        {
            var outRows = (ForwardInputPadding.Rows - FilterRows) / Stride + 1;
            var outColumns = (ForwardInputPadding.Columns - FilterColumns) / Stride + 1;
            return (outRows, outColumns);
        }

        private Tensor3 DoPadding(Tensor3 input)
        {
            if (Padding == 0)
                return input;

            var result = new Tensor3(input.Rows + Padding * 2, input.Columns + Padding * 2, input.Depth);
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
        public Tensor3 Weights { get; private set; }
        public int Rows { get { return Weights.Rows; } }
        public int Columns { get { return Weights.Columns; } }
        public int Depth { get { return Weights.Depth; } }
        public double Bias { get; private set; }

        public Filter(int rows, int columns, int depth)
        {
            Weights = new Tensor3(depth, rows, columns);
        }

        public void SetWeights(Tensor3 weights)
        {
            Weights = weights;
        }

        public void SetBias(double bias)
        {
            Bias = bias;
        }
    }
}
