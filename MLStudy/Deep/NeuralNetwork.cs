/*
 * Description:神经网络
 * Author:YunXiao An
 * Date:2015.11.20
 */


using MLStudy.Abstraction;
using MLStudy.Optimization;
using MLStudy.Regularizations;
using System.Collections.Generic;

namespace MLStudy.Deep
{
    /// <summary>
    /// 神经网络
    /// </summary>
    public sealed class NeuralNetwork : IModel
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

        public IRegularizer Regularizer { get; private set; }

        //记录最近一次训练的输入结构
        private int[] trainInputShape;
        private int[] trainYShape;
        //记录最近一次预测的输入结构
        private int[] predictInputShape;

        /// <summary>
        /// 创建一个神经网络
        /// </summary>
        public NeuralNetwork()
        {
            TrainingLayers = new List<ILayer>();
            PredictLayers = new List<ILayer>();
        }

        public Tensor Predict(Tensor X)
        {
            CheckPredictShape(X);

            for (int i = 0; i < PredictLayers.Count; i++)
            {
                X = PredictLayers[i].Forward(X);
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
            //正则化
            Regularize();
            //更新参数
            Optimize();
        }

        public void PrepareTrain(Tensor X, Tensor y)
        {
            trainInputShape = X.shape;
            trainYShape = y.shape;

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

        public double GetTrainLoss()
        {
            return LossFunction.GetLoss();
        }

        public double GetLoss(Tensor y, Tensor yHat)
        {
            return LossFunction.GetLoss(y, yHat);
        }

        public double GetTrainAccuracy()
        {
            return LossFunction.GetAccuracy();
        }

        public double GetAccuracy(Tensor y, Tensor yHat)
        {
            return LossFunction.GetAccuracy(y, yHat);
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

        public void Regularize()
        {
            if (Regularizer == null)
                return;

            foreach (var item in TrainingLayers)
            {
                if (item is IOptimizable opt)
                    opt.Regularize(Regularizer);
            }
        }

        public NeuralNetwork AddGroup(int groupCount, params ILayer[] layers)
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

        public NeuralNetwork AddFullLayer(int unitCount)
        {
            AddLayer(new FullLayer(unitCount));
            return this;
        }

        public NeuralNetwork AddReLU()
        {
            AddLayer(new ReLU());
            return this;
        }

        public NeuralNetwork AddSigmoid()
        {
            AddLayer(new Sigmoid());
            return this;
        }

        public NeuralNetwork AddSoftmax()
        {
            AddLayer(new Softmax());
            return this;
        }

        public NeuralNetwork AddTanh()
        {
            AddLayer(new Tanh());
            return this;
        }

        public NeuralNetwork UseCrossEntropyLoss()
        {
            UseLossFunction(new CrossEntropy());
            return this;
        }

        public NeuralNetwork UseMeanSquareError()
        {
            UseLossFunction(new MeanSquareError());
            return this;
        }

        public NeuralNetwork UseGradientDescent(double learningRate)
        {
            Optimizer = new GradientDescent(learningRate);
            return this;
        }

        public NeuralNetwork UseAdam(double alpha=0.001, double beta1 = 0.9, double beta2 = 0.999)
        {
            UseOptimizer(new Adam()
            {
                Alpha = alpha,
                Beta1 = beta1,
                Beta2 = beta2,
            });
            return this;
        }

        public NeuralNetwork UseLASSO(double strength)
        {
            UseRegularizer(new Lasso(strength));
            return this;
        }

        public NeuralNetwork UseRidge(double strength)
        {
            UseRegularizer(new Ridge(strength));
            return this;
        }

        public NeuralNetwork AddLayer(ILayer layer)
        {
            TrainingLayers.Add(layer);
            if (layer is IOptimizable opt)
                PredictLayers.Add(opt.CreateMirror());
            else
                PredictLayers.Add(layer.CreateSame());

            return this;
        }

        public NeuralNetwork UseLossFunction(LossFunction loss)
        {
            LossFunction = loss;
            return this;
        }

        public NeuralNetwork UseOptimizer(IOptimizer optimizer)
        {
            Optimizer = optimizer;
            return this;
        }

        public NeuralNetwork UseRegularizer(IRegularizer regularizer)
        {
            Regularizer = regularizer;
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
            if (a == null || b == null)
                return false;

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
