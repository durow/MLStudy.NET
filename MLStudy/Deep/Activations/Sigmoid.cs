/*
 * Description:使用Sigmoid函数的激活层
 * Author:YunXiao An
 * Date:2018.11.19
 */


using MLStudy.Abstraction;

namespace MLStudy.Deep
{
    /// <summary>
    /// 使用Sigmoid函数的激活层
    /// </summary>
    public sealed class Sigmoid : Activation
    {
        /// <summary>
        /// 运行前的准备，用于初始化所有Tensor的结构
        /// </summary>
        /// <param name="input">输入Tensor</param>
        /// <returns></returns>
        public override TensorOld PrepareTrain(TensorOld input)
        {
            ForwardOutput = input.GetSameShape();
            BackwardOutput = input.GetSameShape();
            Derivative = input.GetSameShape();
            return ForwardOutput;
        }

        public override TensorOld PreparePredict(TensorOld input)
        {
            ForwardOutput = input.GetSameShape();
            return ForwardOutput;
        }

        /// <summary>
        /// 正向传播或叫向后传播
        /// </summary>
        /// <param name="input">输入的数值</param>
        /// <returns>输出的数值</returns>
        public override TensorOld Forward(TensorOld input)
        {
            TensorOld.Apply(input, ForwardOutput, Functions.Sigmoid);
            return ForwardOutput;
        }

        /// <summary>
        /// 反向传播或叫向前传播
        /// </summary>
        /// <param name="error">传回来的误差</param>
        /// <returns>传到前面的误差</returns>
        public override TensorOld Backward(TensorOld error)
        {
            TensorOld.Apply(ForwardOutput, Derivative, Derivatives.SigmoidFromOutput);
            TensorOld.MultipleElementWise(Derivative, error, BackwardOutput);
            return BackwardOutput;
        }

        public override ILayer CreateSame()
        {
            return new Sigmoid();
        }
    }
}
