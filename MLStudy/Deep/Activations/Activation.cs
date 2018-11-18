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
        /// 激活函数的名称
        /// </summary>
        public string Name { get; set; }
        
        /// <summary>
        /// 最后一次Forward的输出
        /// </summary>
        public Tensor LastForward { get; private set; }

        /// <summary>
        /// 最后一次Backward向前传播的结果
        /// </summary>
        public Tensor LastBackward { get; set; }

        /// <summary>
        /// 清空LastForwad和LastBackward的缓存
        /// </summary>
        public virtual void ClearCache()
        {
            LastForward = null;
            LastBackward = null;
        }


        public virtual Tensor Prepare(Tensor input)
        {
            LastForward = input.GetSameShape();
            LastBackward = input.GetSameShape();
            return LastBackward;
        }

        /// <summary>
        /// 反向传播或叫向前传播
        /// </summary>
        /// <param name="error">传回来的误差</param>
        /// <returns>传到前面的误差</returns>
        public abstract Tensor Backward(Tensor error);

        /// <summary>
        /// 正向传播或叫向后传播
        /// </summary>
        /// <param name="input">输入的数值</param>
        /// <returns>输出的数值</returns>
        public abstract Tensor Forward(Tensor input);
    }
}
