﻿/*
 * Description:使用交叉熵的损失函数
 * Author:Yunxiao An
 * Date:2018.11.19
 */

using MLStudy.Abstraction;
using System;

namespace MLStudy
{
    /// <summary>
    /// 创建使用交叉熵的损失函数
    /// </summary>
    public sealed class CrossEntropy : LossFunction
    {
        private double[] yBuff;
        private double[] yHatBuff;
        private double[] derBuff;
        private int sampleNumber;

        /// <summary>
        /// 训练前的准备工作，检查并确定所需Tensor的结构并分配好内存
        /// </summary>
        /// <param name="y">样本标签</param>
        /// <param name="yHat">输出标签</param>
        public override void PrepareTrain(Tensor y, Tensor yHat)
        {
            Tensor.CheckShape(y, yHat);
            if (y.Rank != 2)
                throw new TensorShapeException("y and yHat must Rank=2");

            ForwardOutput = new Tensor(y.Shape[0]);
            BackwardOutput = yHat.GetSameShape();
            yBuff = new double[y.shape[1]];
            yHatBuff = new double[y.shape[1]];
            derBuff = new double[y.shape[1]];
            sampleNumber = y.Shape[0];
        }

        /// <summary>
        /// 计算Loss和Loss对yHat的导数（梯度）
        /// </summary>
        /// <param name="y"></param>
        /// <param name="yHat"></param>
        public override void Compute(Tensor y, Tensor yHat)
        {
            var outData = ForwardOutput.GetRawValues();
            var derData = BackwardOutput.GetRawValues();

            for (int i = 0; i < sampleNumber; i++)
            {
                //取出一个样本及其对应的Label
                y.GetByDim1(i, yBuff);
                yHat.GetByDim1(i, yHatBuff);
                //计算交叉熵
                outData[i] = Functions.CrossEntropy(yBuff, yHatBuff);

                //计算损失函数关于输入的导数
                Derivatives.CrossEntropy(yBuff, yHatBuff, derBuff);
                Array.Copy(derBuff, 0, derData, i * derBuff.Length, derBuff.Length);
            }
        }

        public override double GetLoss(Tensor y, Tensor yHat)
        {
            var outData = ForwardOutput.GetRawValues();
            var derData = BackwardOutput.GetRawValues();

            var result = 0d;
            for (int i = 0; i < sampleNumber; i++)
            {
                //取出一个样本及其对应的Label
                y.GetByDim1(i, yBuff);
                yHat.GetByDim1(i, yHatBuff);
                //计算交叉熵
                result += Functions.CrossEntropy(yBuff, yHatBuff);
            }
            return result / sampleNumber;
        }
    }
}