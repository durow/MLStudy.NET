/*
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
    public sealed class CrossEntropyFromSoftmax : LossFunction
    {
        private double[] yBuff;
        private double[] yHatBuff;
        private double[] derBuff;
        private int sampleNumber;
        private TensorOld LastY;
        private TensorOld LastYHat;

        /// <summary>
        /// 训练前的准备工作，检查并确定所需Tensor的结构并分配好内存
        /// </summary>
        /// <param name="y">样本标签</param>
        /// <param name="yHat">输出标签</param>
        public override void PrepareTrain(TensorOld y, TensorOld yHat)
        {
            TensorOld.CheckShape(y, yHat);
            if (y.Rank != 2)
                throw new TensorShapeException("y and yHat must Rank=2");

            ForwardOutput = new TensorOld(y.shape[0]);
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
        public override void Compute(TensorOld y, TensorOld yHat)
        {
            LastY = y;
            LastYHat = yHat;
            ComputeCrossEntropy(y, yHat);
        }

        public override double  GetLoss(TensorOld y, TensorOld yHat)
        {
            var outData = ForwardOutput.GetRawValues();

            var result = 0d;
            for (int i = 0; i < y.shape[0]; i++)
            {
                //取出一个样本及其对应的Label
                y.GetByDim1(i, yBuff);
                yHat.GetByDim1(i, yHatBuff);
                //计算交叉熵
                result += Functions.CrossEntropy(yBuff, yHatBuff);
            }

            return result / sampleNumber;
        }

        public override double GetAccuracy()
        {
            return GetAccuracy(LastY, LastYHat);
        }

        public override double GetAccuracy(TensorOld y, TensorOld yHat)
        {
            var code = Utilities.ProbabilityToCode(yHat);
            return ComputeAccuracy(y, code);
        }

        private void ComputeCrossEntropy(TensorOld y, TensorOld yHat)
        {
            var foreoutData = ForwardOutput.GetRawValues();

            for (int i = 0; i < sampleNumber; i++)
            {
                //取出一个样本及其对应的Label
                y.GetByDim1(i, yBuff);
                yHat.GetByDim1(i, yHatBuff);
                //计算交叉熵
                foreoutData[i] = Functions.CrossEntropy(yBuff, yHatBuff);
            }

            Array.Copy(y.values, 0, BackwardOutput.values, 0, y.ElementCount);
        }

        public static double ComputeAccuracy(TensorOld y, TensorOld yHat)
        {
            var count = y.shape[0];
            var eq = 0d;
            for (int i = 0; i < count; i++)
            {
                if (RowEqual(y, yHat, i * y.shape[1], y.shape[1]))
                    eq++;
            }
            return eq / count;
        }

        private static bool RowEqual(TensorOld t1, TensorOld t2, int start, int len)
        {
            for (int i = 0; i < len; i++)
            {
                if (t1.GetRawValues()[start + i] != t2.GetRawValues()[start + i])
                    return false;
            }
            return true;
        }
    }
}
