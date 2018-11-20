using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace MLStudy.Deep
{
    public sealed class FullLayer : ILayer, IOptimizable
    {
        public string Name { get; set; }
        public Tensor ForwardInput { get; private set; } //对之前输入的引用，为了方便计算权重的梯度，不需要初始化
        public Tensor ForwardOutput { get; private set; }
        public Tensor BackwardOutput { get; private set; }
        public Tensor Weights { get; private set; }
        public Tensor Bias { get; private set; }
        public Tensor WeightsGradient { get; private set; }
        public Tensor BiasGradient { get; private set; }
        private Tensor gradientBuff;

        public int UnitCount { get; set; }

        public FullLayer(int unitCount)
        {
            UnitCount = unitCount;
        }

        public Tensor PrepareTrain(Tensor input)
        {
            if (input.Rank != 2)
                throw new TensorShapeException("input tensor must have Rank=2");

            ForwardOutput = new Tensor(input.Shape[0], UnitCount);
            BackwardOutput = input.GetSameShape();
            Weights = new Tensor(input.Shape[1], UnitCount);
            WeightsGradient = Weights.GetSameShape();
            BiasGradient = Bias.GetSameShape();
            Bias = new Tensor(1, UnitCount);
            gradientBuff = WeightsGradient.GetSameShape();

            return ForwardOutput;
        }

        public Tensor Forward(Tensor input)
        {
            //保存input用于Backward中计算Weights和Bias的梯度
            ForwardInput = input;
            Tensor.Multiple(input, Weights, ForwardOutput);
            AddBias();
            return ForwardOutput;
        }

        public Tensor Backward(Tensor error)
        {
            Tensor.Multiple(error, Weights.Transpose(), BackwardOutput);
            return BackwardOutput;
        }

        public void Optimize(IOptimizer optimizer)
        {
            optimizer.Optimize(Weights, WeightsGradient);
            optimizer.Optimize(Bias, BiasGradient);
        }

        private void AddBias()
        {
            var outData = ForwardOutput.GetRawValues();
            var biasData = Bias.GetRawValues();

            Parallel.For(0, ForwardOutput.shape[0], i =>
            {
                var start = i * UnitCount;
                for (int j = 0; i < UnitCount; i++)
                {
                    outData[start + j] += biasData[j];
                }
            });
        }
    }
}
