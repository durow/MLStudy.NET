/*
 * Description:最普通的梯度下降
 * Author:YunXiao An
 * Date:2018.11.20
 */


using MLStudy.Abstraction;

namespace MLStudy
{
    /// <summary>
    /// 最基本的梯度下降优化器
    /// </summary>
    public sealed class GradientDescent : IOptimizer
    {
        /// <summary>
        /// 学习速率
        /// </summary>
        public double LearningRate { get; set; }

        /// <summary>
        /// 创建一个最基本的梯度下降优化器
        /// </summary>
        /// <param name="learningRate">学习速率</param>
        public GradientDescent(double learningRate)
        {
            LearningRate = learningRate;
        }

        /// <summary>
        /// 根据gradient优化target
        /// </summary>
        /// <param name="target">优化的目标</param>
        /// <param name="graident">目标的梯度</param>
        public void Optimize(TensorOld target, TensorOld graident)
        {
            TensorOld.Apply(target, graident, target, (a, b) => a - LearningRate * b);
        }
    }
}
