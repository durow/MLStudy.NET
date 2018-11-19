/*
 * Description:使用Softmax函数的激活层
 * Author:YunXiao An
 * Date:2018.11.19
 */


using System;
using System.Threading.Tasks;

namespace MLStudy.Deep
{
    /// <summary>
    /// 使用Softmax函数的激活层
    /// </summary>
    public class Softmax : Activations.Activation
    {
        private double[] sampleBuff;
        private int sampleNumber;
        private int categoryNumber;

        /// <summary>
        /// 运行前的准备，用于初始化所有Tensor的结构
        /// </summary>
        /// <param name="input">输入Tensor</param>
        /// <returns></returns>
        public override Tensor PrepareTrain(Tensor input)
        {
            if (input.Rank != 2)
                throw new TensorShapeException("input.Rank must be 2");

            ForwardOutput = input.GetSameShape();
            BackwardOutput = input.GetSameShape();
            sampleNumber = input.Shape[0];
            categoryNumber = input.Shape[1];
            sampleBuff = new double[categoryNumber];
            //向量对向量求导的结果是个矩阵
            //多个样本下，softmax的导数是一个三阶张量，第一维是样本数量，后面两维是jacob矩阵
            Derivative = new Tensor(sampleNumber, categoryNumber, categoryNumber);
            return ForwardOutput;
        }

        /// <summary>
        /// 正向传播或叫向后传播
        /// </summary>
        /// <param name="input">输入的数值</param>
        /// <returns>输出的数值</returns>
        public override Tensor Forward(Tensor input)
        {
            var src = input.GetRawValues();
            var des = ForwardOutput.GetRawValues();
            int startIndex;

            for (int i = 0; i < sampleNumber; i++)
            {
                startIndex = categoryNumber * i;
                Array.Copy(src, startIndex, sampleBuff, 0, categoryNumber);
                Functions.Softmax(sampleBuff, sampleBuff);
                Array.Copy(sampleBuff, 0, des, startIndex, categoryNumber);
            }
            return ForwardOutput;
        }

        /// <summary>
        /// 反向传播或叫向前传播
        /// </summary>
        /// <param name="error">传回来的误差</param>
        /// <returns>传到前面的误差</returns>
        public override Tensor Backward(Tensor error)
        {
            ComputeDerivative();
            Parallel.For(0, sampleNumber, i =>
            {
                //这个方法直接将计算结果写入result，不需要开辟中间内存
                ErrorBP(ForwardOutput, error, BackwardOutput, i);
            });

            return BackwardOutput;
        }

        //这个方法不会产生多余的临时对象，问题就是不再存储Derivative
        private void ErrorBP(Tensor output, Tensor error, Tensor result, int sampleIndex)
        {
            for (int i = 0; i < categoryNumber; i++)
            {
                var der = 0d;
                for (int j = 0; j < categoryNumber; j++)
                {
                    if (i == j)
                        der += output[sampleIndex, i] * (1 - output[sampleIndex, j]) * error[sampleIndex, j];
                    else
                        der += -output[sampleIndex, i] * output[sampleIndex, j] * error[sampleIndex, j];
                }
                result[sampleIndex, i] = der;
            }
        }

        private void ErrorBP(Tensor error)
        {
            var derData = Derivative.GetRawValues();
            var errorData = error.GetRawValues();
            var outData = BackwardOutput.GetRawValues();

            Parallel.For(0, sampleNumber, sampleIndex =>
            {
                var errorStart = error.GetRawOffset(sampleIndex, 0);
                //这里的两层嵌套执行的并不是严格的矩阵运算，导数应该是：error*jacob，
                //因为jacob矩阵是对称的所以使用jacob每行和error相乘的内积，循环写起来方便
                Parallel.For(0, categoryNumber, i =>
                {
                    var derStart = Derivative.GetRawOffset(sampleNumber, i, 0);
                    var sum = 0d;
                    for (int j = 0; j < categoryNumber; j++)
                    {
                        sum += derData[derStart + j] * errorData[errorStart + j];
                    }
                    outData[errorStart + i] = sum;
                });
            });
        }

        private void ComputeDerivative()
        {
            var derData = Derivative.GetRawValues();
            var outData = ForwardOutput.GetRawValues();
            Parallel.For(0, sampleNumber, sampleIndex =>
            {
                var outStart = ForwardOutput.GetRawOffset(sampleIndex, 0);
                Parallel.For(0, categoryNumber, i =>
                {
                    var derStart = Derivative.GetRawOffset(sampleIndex, i, 0);
                    for (int j = 0; j < categoryNumber; j++)
                    {
                        if (i == j)
                            derData[derStart + j] = outData[outStart + i] * (1 - outData[outStart + j]);
                        else
                            derData[derStart + j] = -outData[outStart + i] * outData[outStart + j];
                    }
                });
            });
        }
    }
}
