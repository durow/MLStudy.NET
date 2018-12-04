using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public partial class TensorOld
    {
        
        public static TensorOld Rand()
        {
            return new TensorOld();
        }
        /// <summary>
        /// 创建一个由shape指定结构的Tensor并用0-1的随机数填充
        /// </summary>
        /// <param name="shape">Tensor的结构</param>
        /// <returns>创建好的Tensor</returns>
        public static TensorOld Rand(params int[] shape)
        {
            var len = GetTotalLength(shape);
            var data = DataEmulator.Instance.RandomArray(len);
            return new TensorOld(data, shape);
        }

        public static TensorOld RandGaussian()
        {
            return new TensorOld();
        }

        /// <summary>
        /// 创建一个由shape指定结构的Tensor并用符合N(0,1)高斯分布的数值填充
        /// </summary>
        /// <param name="shape">Tensor的结构</param>
        /// <returns>创建好的Tensor</returns>
        public static TensorOld RandGaussian(params int[] shape)
        {
            var len = GetTotalLength(shape);
            var data = DataEmulator.Instance.RandomArrayGaussian(len);
            return new TensorOld(data, shape);
        }

        /// <summary>
        /// 创建一个由shape指定结构的Tensor并用0填充
        /// </summary>
        /// <param name="shape">Tensor的结构</param>
        /// <returns>创建好的Tensor</returns>
        public static TensorOld Zeros(params int[] shape)
        {
            return new TensorOld(shape);
        }

        /// <summary>
        /// 创建一个由shape指定结构的Tensor并用1填充
        /// </summary>
        /// <param name="shape">Tensor的结构</param>
        /// <returns>创建好的Tensor</returns>
        public static TensorOld Ones(params int[] shape)
        {
            return Values(1, shape);
        }

        public static TensorOld Values(double value)
        {
            var result = new TensorOld();
            result.SetValue(value);
            return result;
        }

        /// <summary>
        /// 创建一个由shape指定结构的Tensor并用参数value的值填充
        /// </summary>
        /// <param name="value">填充的值</param>
        /// <param name="shape">Tensor的结构</param>
        /// <returns>创建好的Tensor</returns>
        public static TensorOld Values(double value, params int[] shape)
        {
            var len = GetTotalLength(shape);
            var data = new double[len];
            for (int i = 0; i < len; i++)
            {
                data[i] = value;
            }
            return new TensorOld(data, shape);
        }

        /// <summary>
        /// 创建一个二维的单位矩阵
        /// </summary>
        /// <param name="width">矩阵的宽度和高度</param>
        /// <returns></returns>
        public static TensorOld I(int width)
        {
            var result = new TensorOld(width, width);
            for (int i = 0; i < width; i++)
            {
                result.SetValueFast(1, i, i);
            }
            return result;
        }
    }
}
