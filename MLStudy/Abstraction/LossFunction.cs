/*
 * Description:损失函数的抽象基类，定义了损失函数的一些基本操作
 * Author:Yunxiao An
 * Date:2018.11.19
 */

namespace MLStudy.Abstraction
{
    public abstract class LossFunction
    {
        /// <summary>
        /// 输出的损失，一阶张量（向量），每个分量对应着一个样本的Loss。
        /// 使用GetLoss()方法输出这些样本的平均Loss
        /// </summary>
        public TensorOld ForwardOutput { get; protected set; }

        /// <summary>
        /// 反向传播的Loss的导数或叫梯度
        /// </summary>
        public TensorOld BackwardOutput { get; protected set; }

        /// <summary>
        /// 获取最近一次Forward的所有样本Loss的平均值。
        /// 通过ForwardOutput可以获取每个样本的Loss
        /// </summary>
        /// <returns>Loss平均值</returns>
        public double GetLoss()
        {
            return ForwardOutput.Mean();
        }

        public abstract double GetAccuracy();

        /// <summary>
        /// 模型预测后可以使用这个函数计算Loss
        /// </summary>
        /// <param name="y">真实值</param>
        /// <param name="yHat">预测值</param>
        /// <returns>Loss</returns>
        public abstract double GetLoss(TensorOld y, TensorOld yHat);

        public abstract double GetAccuracy(TensorOld y, TensorOld yHat);

        /// <summary>
        /// 训练前的准备工作，检查并确定所需Tensor的结构并分配好内存
        /// </summary>
        /// <param name="y">样本标签</param>
        /// <param name="yHat">输出标签</param>
        public abstract void PrepareTrain(TensorOld y, TensorOld yHat);

        /// <summary>
        /// 计算Loss和Loss对yHat的导数（梯度）
        /// </summary>
        /// <param name="y"></param>
        /// <param name="yHat"></param>
        public abstract void Compute(TensorOld y, TensorOld yHat);
    }
}
