/*
 * Description:激活函数的抽象基类
 * Author:YunXiao An
 * Date:2018.11.18
 */

using MLStudy.Abstraction;

namespace MLStudy.Deep.Activations
{
    public abstract class Activation : ILayer
    {
        /// <summary>
        /// 激活层的名称
        /// </summary>
        public string Name { get; set; }
        
        /// <summary>
        /// 最后一次Forward的输出
        /// </summary>
        public Tensor ForwardOutput { get; protected set; }

        /// <summary>
        /// 最后一次Backward向前传播的结果
        /// </summary>
        public Tensor BackwardOutput { get; protected set; }

        /// <summary>
        /// 反向传播的梯度
        /// </summary>
        public Tensor Derivative { get; protected set; }

        /// <summary>
        /// 清空LastForwad和LastBackward的缓存
        /// </summary>
        public virtual void ClearCache()
        {
            ForwardOutput = null;
            BackwardOutput = null;
            Derivative = null;
        }


        /// <summary>
        /// 运行前的准备，用于初始化所有Tensor的结构
        /// </summary>
        /// <param name="input">输入Tensor</param>
        /// <returns></returns>
        public abstract Tensor PrepareTrain(Tensor input);

        /// <summary>
        /// 正向传播或叫向后传播
        /// </summary>
        /// <param name="input">输入的数值</param>
        /// <returns>输出的数值</returns>
        public abstract Tensor Forward(Tensor input);

        /// <summary>
        /// 反向传播或叫向前传播
        /// </summary>
        /// <param name="error">传回来的误差</param>
        /// <returns>传到前面的误差</returns>
        public abstract Tensor Backward(Tensor error);

        /// <summary>
        /// 用于预测时的准备工作
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public abstract Tensor PreparePredict(Tensor input);

        /// <summary>
        /// 创建一个同样结构的层
        /// </summary>
        /// <returns></returns>
        public abstract ILayer CreateSame();
    }
}
