using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public partial class Tensor
    {
        
        public static Tensor Rand()
        {
            return new Tensor();
        }
        /// <summary>
        /// 创建一个由shape指定结构的Tensor并用0-1的随机数填充
        /// </summary>
        /// <param name="shape">Tensor的结构</param>
        /// <returns>创建好的Tensor</returns>
        public static Tensor Rand(params int[] shape)
        {
            var len = GetTotalLength(shape);
            var data = DataEmulator.Instance.RandomArray(len);
            return new Tensor(data, shape);
        }

        public static Tensor RandGaussian()
        {
            return new Tensor();
        }

        /// <summary>
        /// 创建一个由shape指定结构的Tensor并用符合N(0,1)高斯分布的数值填充
        /// </summary>
        /// <param name="shape">Tensor的结构</param>
        /// <returns>创建好的Tensor</returns>
        public static Tensor RandGaussian(params int[] shape)
        {
            var len = GetTotalLength(shape);
            var data = DataEmulator.Instance.RandomArrayGaussian(len);
            return new Tensor(data, shape);
        }

        /// <summary>
        /// 创建一个由shape指定结构的Tensor并用0填充
        /// </summary>
        /// <param name="shape">Tensor的结构</param>
        /// <returns>创建好的Tensor</returns>
        public static Tensor Zeros(params int[] shape)
        {
            return new Tensor(shape);
        }

        /// <summary>
        /// 创建一个由shape指定结构的Tensor并用1填充
        /// </summary>
        /// <param name="shape">Tensor的结构</param>
        /// <returns>创建好的Tensor</returns>
        public static Tensor Ones(params int[] shape)
        {
            return Values(1, shape);
        }

        public static Tensor Values(double value)
        {
            var result = new Tensor();
            result.SetValue(value);
            return result;
        }

        /// <summary>
        /// 创建一个由shape指定结构的Tensor并用参数value的值填充
        /// </summary>
        /// <param name="value">填充的值</param>
        /// <param name="shape">Tensor的结构</param>
        /// <returns>创建好的Tensor</returns>
        public static Tensor Values(double value, params int[] shape)
        {
            var len = GetTotalLength(shape);
            var data = new double[len];
            for (int i = 0; i < len; i++)
            {
                data[i] = value;
            }
            return new Tensor(data, shape);
        }

        /// <summary>
        /// 创建一个二维的单位矩阵
        /// </summary>
        /// <param name="width">矩阵的宽度和高度</param>
        /// <returns></returns>
        public static Tensor I(int width)
        {
            var result = new Tensor(width, width);
            for (int i = 0; i < width; i++)
            {
                result.SetValueFast(1, i, i);
            }
            return result;
        }
    }
}
