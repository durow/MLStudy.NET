/*
 * Desctiption:全连接层
 * Author:YunXiao An
 * Date:2018.11.20
 */


using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace MLStudy.Deep
{
    /// <summary>
    /// 全连接层
    /// </summary>
    public sealed class FullLayer : ILayer, IOptimizable
    {
        /// <summary>
        /// 名称
        /// </summary>
        public string Name { get; set; }
        /// <summary>
        /// 输入的引用
        /// </summary>
        public Tensor ForwardInput { get; private set; } //对之前输入的引用，为了方便计算权重的梯度，不需要初始化
        /// <summary>
        /// 最后一次正向传播的输出
        /// </summary>
        public Tensor ForwardOutput { get; private set; }
        /// <summary>
        /// 最后一次反向传播的输出
        /// </summary>
        public Tensor BackwardOutput { get; private set; }
        /// <summary>
        /// 权重，每列代表一个Unit
        /// </summary>
        public Tensor Weights { get; private set; }
        /// <summary>
        /// 偏置，Shape为1*N，每一个对应一个Unit
        /// </summary>
        public Tensor Bias { get; private set; }
        /// <summary>
        /// 反向传播计算出的权重的梯度
        /// </summary>
        public Tensor WeightsGradient { get; private set; }
        /// <summary>
        /// 反向传播计算出的偏置的梯度
        /// </summary>
        public Tensor BiasGradient { get; private set; }
        /// <summary>
        /// 单元个数
        /// </summary>
        public int UnitCount { get; set; }

        private List<FullLayer> mirrorList = new List<FullLayer>();
        private int[] sampleStartIndex;
        private int[] errorStartIndex;

        /// <summary>
        /// 创建一个包含unitCount个单元的全连接层
        /// </summary>
        /// <param name="unitCount"></param>
        public FullLayer(int unitCount)
        {
            UnitCount = unitCount;
            mirrorList.Add(this);
        }

        /// <summary>
        /// 训练前的准备工作，检查结构，分配内存等
        /// </summary>
        /// <param name="input">输入，主要使用结构信息</param>
        /// <returns></returns>
        public Tensor PrepareTrain(Tensor input)
        {
            if (input.Rank != 2)
                throw new TensorShapeException("input tensor must have Rank=2");

            ForwardOutput = new Tensor(input.shape[0], UnitCount);
            BackwardOutput = input.GetSameShape();

            if(Weights == null)
            {
                SetWeights(Tensor.RandGaussian(input.shape[1], UnitCount));
                WeightsGradient = Weights.GetSameShape();
            }
            else
            {
                if (input.shape[1] != Weights.shape[0])
                    throw new TensorShapeException("input and weights are not match!");
            }

            if (Bias == null)
            {
                SetBias(Tensor.Zeros(1, UnitCount));
                BiasGradient = Bias.GetSameShape();
            }

            sampleStartIndex = new int[input.shape[0]];
            errorStartIndex = new int[input.shape[0]];
            for (int i = 0; i < sampleStartIndex.Length; i++)
            {
                sampleStartIndex[i] = i * input.shape[1];
                errorStartIndex[i] = i * UnitCount;
            }

            return ForwardOutput;
        }

        /// <summary>
        /// 用于预测时的内存分配
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public Tensor PreparePredict(Tensor input)
        {
            if (input.Rank != 2)
                throw new TensorShapeException("input tensor must have Rank=2");

            ForwardOutput = new Tensor(input.shape[0], UnitCount);
            return ForwardOutput;
        }

        /// <summary>
        /// 正向传播
        /// </summary>
        /// <param name="input">输入</param>
        /// <returns>输出</returns>
        public Tensor Forward(Tensor input)
        {
            //保存input用于Backward中计算Weights和Bias的梯度
            ForwardInput = input;
            Tensor.Multiple(input, Weights, ForwardOutput);
            AddBias();
            return ForwardOutput;
        }

        /// <summary>
        /// 反向传播
        /// </summary>
        /// <param name="error">输入</param>
        /// <returns>输出</returns>
        public Tensor Backward(Tensor error)
        {
            //计算Weights和Bias的梯度
            ComputeGradient(error);
            //反向传播的误差
            //var trans = Weights.Transpose();
            //Tensor.Multiple(error, trans, BackwardOutput);

            Parallel.For(0, error.shape[0], i =>
            {
                Parallel.For(0, Weights.shape[0], j =>
                {
                    var sum = 0d;
                    for (int k = 0; k < error.shape[1]; k++)
                    {
                        sum += error[i, k] * Weights[j, k];
                    }
                    BackwardOutput[i, j] = sum;
                });
            });

            return BackwardOutput;
        }

        /// <summary>
        /// 优化权重和偏置
        /// </summary>
        /// <param name="optimizer"></param>
        public void Optimize(IOptimizer optimizer)
        {
            optimizer.Optimize(Weights, WeightsGradient);
            optimizer.Optimize(Bias, BiasGradient);
        }

        public void Regularize(IRegularizer regularizer)
        {
            regularizer.Regularize(Weights, WeightsGradient);
        }

        /// <summary>
        /// 手动设置权重
        /// </summary>
        /// <param name="weights"></param>
        public void SetWeights(Tensor weights)
        {
            if(Weights == null)
            {
                foreach (var item in mirrorList)
                {
                    item.Weights = weights;
                }
            }
            Tensor.CheckShape(weights, Weights);
            Array.Copy(weights.GetRawValues(), 0, Weights.GetRawValues(), 0, Weights.ElementCount);
        }

        /// <summary>
        /// 手动设置偏置
        /// </summary>
        /// <param name="bias"></param>
        public void SetBias(Tensor bias)
        {
            if (Bias == null)
            {
                foreach (var item in mirrorList)
                {
                    item.Bias = bias;
                }
            }
            Tensor.CheckShape(Bias, bias);
            Array.Copy(bias.GetRawValues(), 0, Bias.GetRawValues(), 0, Bias.ElementCount);
        }

        /// <summary>
        /// 获取一个共享可优化部分的镜像
        /// </summary>
        /// <returns></returns>
        public ILayer CreateMirror()
        {
            var result = new FullLayer(UnitCount);
            result.Weights = Weights;
            result.Bias = Bias;
            mirrorList.Add(result);
            result.mirrorList = mirrorList;
            return result;
        }

        /// <summary>
        /// 创建一个同样类型的层
        /// </summary>
        /// <returns></returns>
        public ILayer CreateSame()
        {
            return new FullLayer(UnitCount);
        }

        //正向传播输出加上偏置
        private void AddBias()
        {
            var outData = ForwardOutput.GetRawValues();
            var biasData = Bias.GetRawValues();

            Parallel.For(0, ForwardOutput.shape[0], i =>
            {
                var start = i * UnitCount;
                for (int j = 0; j < UnitCount; j++)
                {
                    outData[start + j] += biasData[j];
                }
            });
        }

        //计算Weights和Bias的梯度
        private void ComputeGradient(Tensor error)
        {
            var inputData = ForwardInput.GetRawValues();
            var errorData = error.GetRawValues();

            Parallel.For(0, WeightsGradient.shape[0], i =>
            {
                Parallel.For(0, UnitCount, j =>
                {
                    var weightSum = 0d;
                    var biasSum = 0d;
                    for (int k = 0; k < sampleStartIndex.Length; k++)
                    {
                        weightSum += inputData[sampleStartIndex[k] + i] * errorData[errorStartIndex[k] + j];
                        biasSum += errorData[errorStartIndex[k] + j];
                    }
                    WeightsGradient[i, j] = weightSum;
                    BiasGradient.GetRawValues()[j] = biasSum;
                });
            });
        }

    }
}
