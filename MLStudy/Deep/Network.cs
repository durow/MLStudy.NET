/*
 * Description:神经网络
 * Author:YunXiao An
 * Date:2015.11.20
 */


using MLStudy.Abstraction;
using System.Collections.Generic;

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
        private int[] trainYShape;
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

        public Tensor Predict(Tensor X)
        {
            CheckPredictShape(X);

            for (int i = 0; i < PredictLayers.Count; i++)
            {
                X = TrainingLayers[i].Forward(X);
            }
            return X;
        }

        public void Step(Tensor X, Tensor y)
        {
            CheckTrainShape(X, y);

            //正向传播
            var yHat = Forward(X);
            //计算Loss
            LossFunction.Compute(y, yHat);
            //反向传播
            Backward(LossFunction.BackwardOutput);
            //更新参数
            Optimize();
        }

        public void PrepareTrain(Tensor X, Tensor y)
        {
            trainInputShape = X.shape;

            for (int i = 0; i < TrainingLayers.Count; i++)
            {
                X = TrainingLayers[i].PrepareTrain(X);
            }

            LossFunction.PrepareTrain(X, y);
        }

        public void PreparePredict(Tensor X)
        {
            predictInputShape = X.shape;

            for (int i = 0; i < PredictLayers.Count; i++)
            {
                X = PredictLayers[i].PreparePredict(X);
            }
        }

        private Tensor Forward(Tensor input)
        {
            for (int i = 0; i < TrainingLayers.Count; i++)
            {
                input = TrainingLayers[i].Forward(input);
            }
            return input;
        }

        private Tensor Backward(Tensor error)
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

        public Network AddGroup(int groupCount, params ILayer[] layers)
        {
            if (groupCount < 1)
                return this;

            for (int i = 0; i < layers.Length; i++)
            {
                AddLayer(layers[i]);
            }

            for (int i = 1; i < groupCount; i++)
            {
                for (int j = 0; j < layers.Length; j++)
                {
                    AddLayer(layers[i].CreateSame());
                }
            }

            return this;
        }

        public Network AddFullLayer(int unitCount)
        {
            AddLayer(new FullLayer(unitCount));
            return this;
        }

        public Network AddReLU()
        {
            AddLayer(new ReLU());
            return this;
        }

        public Network AddSigmoid()
        {
            AddLayer(new Sigmoid());
            return this;
        }

        public Network AddSoftmax()
        {
            AddLayer(new Softmax());
            return this;
        }

        public Network AddTanh()
        {
            AddLayer(new Tanh());
            return this;
        }

        public Network UseCrossEntropyLoss()
        {
            UseLossFunction(new CrossEntropy());
            return this;
        }

        public Network UseMeanSquareError()
        {
            UseLossFunction(new MeanSquareError());
            return this;
        }

        public Network UseGradientDescent(double learningRate)
        {
            Optimizer = new GradientDescent(learningRate);
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

        private void CheckTrainShape(Tensor input, Tensor y)
        {
            if (CheckShape(input.shape, trainInputShape) &&
                CheckShape(y.shape, trainYShape))
                return;

            PrepareTrain(input, y);
        }

        private void CheckPredictShape(Tensor input)
        {
            if (CheckShape(input.shape, predictInputShape))
                return;

            PreparePredict(input);
        }

        private bool CheckShape(int[] a, int[] b)
        {
            if (a.Length != b.Length)
                return false;

            for (int i = 0; i < a.Length; i++)
            {
                if (a[i] != b[i])
                    return false;
            }

            return true;
        }
    }
}
