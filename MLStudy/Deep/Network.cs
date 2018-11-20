using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Deep
{
    /// <summary>
    /// 神经网络
    /// </summary>
    public sealed class Network
    {
        /// <summary>
        /// 用于训练的层
        /// </summary>
        public List<ILayer> TrainingLayers { get; private set; }

        /// <summary>
        /// 用于预测的层
        /// </summary>
        public List<ILayer> PredictLayers { get; private set; }

        /// <summary>
        /// 损失函数
        /// </summary>
        public LossFunction LossFunction { get; private set; }

        /// <summary>
        /// 优化器
        /// </summary>
        public IOptimizer Optimizer { get; private set; }

        //记录最近一次训练的输入结构
        private int[] trainInputShape;
        //记录最近一次预测的输入结构
        private int[] predictInputShape;

        /// <summary>
        /// 创建一个神经网络
        /// </summary>
        public Network()
        {
            TrainingLayers = new List<ILayer>();
            PredictLayers = new List<ILayer>();
        }

        public Tensor Predict(Tensor input)
        {
            throw new NotImplementedException();
        }

        public void Step(Tensor X, Tensor y)
        {
            var yHat = Forward(X);
            LossFunction.Compute(y, yHat);
            Backward(LossFunction.BackwardOutput);
        }

        public Tensor Forward(Tensor input)
        {
            for (int i = 0; i < TrainingLayers.Count; i++)
            {
                input = TrainingLayers[i].Forward(input);
            }
            return input;
        }

        public Tensor Backward(Tensor error)
        {
            var layers = TrainingLayers.Count - 1;
            for (int i = layers; i > -1; i--)
            {
                error = TrainingLayers[i].Backward(error);
            }
            return error;
        }

        public void Optimize()
        {
            foreach (var item in TrainingLayers)
            {
                if (item is IOptimizable opt)
                    opt.Optimize(Optimizer);
            }
        }

        public Network AddFullLayers(params int[] layers)
        {
            return this;
        }

        public Network AddReLU()
        {
            AddLayer(new ReLU());
            return this;
        }

        public Network AddFullLayer(int unitCount)
        {
            return this;
        }

        public Network AddLayer(ILayer layer)
        {
            TrainingLayers.Add(layer);
            if (layer is IOptimizable opt)
                PredictLayers.Add(opt.CreateMirror());
            else
                PredictLayers.Add(layer.CreateSame());

            return this;
        }

        public Network UseLossFunction(LossFunction loss)
        {
            LossFunction = loss;
            return this;
        }

        public Network UseOptimizer(IOptimizer optimizer)
        {
            Optimizer = optimizer;
            return this;
        }
    }
}
