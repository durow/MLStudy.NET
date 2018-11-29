/*
 * Description:张量相关操作，实现了深度学习中用到的大多数计算，加减乘除和转置等。
 *             通过在一维数组的基础上加了一层中间件，用来映射张量的结构和相关操作。
 *             部分张量计算使用了Parallel进行了并行化。
 * Author:Yunxiao An
 * Date:2018.11.16
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MLStudy
{
    public partial class Tensor
    {
        public double this[int d1]
        {
            get
            {
                return GetValue(d1);
            }
            set
            {
                SetValue(value, d1);
            }
        }

        public double this[int d1, int d2]
        {
            get
            {
                return GetValue(d1, d2);
            }
            set
            {
                SetValue(value, d1, d2);
            }
        }

        public double this[int d1, int d2, int d3]
        {
            get
            {
                return GetValue(d1, d2, d3);
            }
            set
            {
                SetValue(value, d1, d2, d3);
            }
        }

        public double this[int d1, int d2, int d3, int d4]
        {
            get
            {
                return GetValue(d1, d2, d3, d4);
            }
            set
            {
                SetValue(value, d1, d2, d3, d4);
            }
        }

        public double this[int d1, int d2, int d3, int d4, int d5]
        {
            get
            {
                return GetValue(d1, d2, d3, d4, d5);
            }
            set
            {
                SetValue(value, d1, d2, d3, d4, d5);
            }
        }

        public double this[int d1, int d2, int d3, int d4, int d5, int d6]
        {
            get
            {
                return GetValue(d1, d2, d3, d4, d5, d6);
            }
            set
            {
                SetValue(value, d1, d2, d3, d4, d5, d6);
            }
        }

        public double this[int d1, int d2, int d3, int d4, int d5, int d6, int d7]
        {
            get
            {
                return GetValue(d1, d2, d3, d4, d5, d6, d7);
            }
            set
            {
                SetValue(value, d1, d2, d3, d4, d5, d6, d7);
            }
        }

        public double this[int d1, int d2, int d3, int d4, int d5, int d6, int d7, int d8]
        {
            get
            {
                return GetValue(d1, d2, d3, d4, d5, d6, d7, d8);
            }
            set
            {
                SetValue(value, d1, d2, d3, d4, d5, d6, d7, d8);
            }
        }

        public double this[int d1, int d2, int d3, int d4, int d5, int d6, int d7, int d8, int d9]
        {
            get
            {
                return GetValue(d1, d2, d3, d4, d5, d6, d7, d8, d9);
            }
            set
            {
                SetValue(value, d1, d2, d3, d4, d5, d6, d7, d8, d9);
            }
        }


        public double GetValueFast()
        {
            return values[0];
        }

        public double GetValueFast(int d1)
        {
            return values[d1];
        }

        public double GetValueFast(int d1, int d2)
        {
            return values[d1 * dimensionSize[0] + d2];
        }

        public double GetValueFast(int d1, int d2, int d3)
        {
            return values[d1 * dimensionSize[0] + d2 * dimensionSize[1] + d3];
        }

        public double GetValueFast(int d1, int d2, int d3, int d4)
        {
            return values[d1 * dimensionSize[0] + d2 * dimensionSize[1] + d3 * dimensionSize[2] + d4];
        }

        public double GetValueFast(int d1, int d2, int d3, int d4, int d5)
        {
            return values[d1 * dimensionSize[0] + d2 * dimensionSize[1] + d3 * dimensionSize[2] + d4 * dimensionSize[3] + d5];
        }

        public double GetValueFast(int d1, int d2, int d3, int d4, int d5, int d6)
        {
            return values[d1 * dimensionSize[0] + 
                d2 * dimensionSize[1] + 
                d3 * dimensionSize[2] + 
                d4 * dimensionSize[3] + 
                d5 * dimensionSize[4] + 
                d6];
        }

        public double GetValueFast(int d1, int d2, int d3, int d4, int d5, int d6, int d7)
        {
            return values[d1 * dimensionSize[0] +
                d2 * dimensionSize[1] +
                d3 * dimensionSize[2] +
                d4 * dimensionSize[3] +
                d5 * dimensionSize[4] +
                d6 * dimensionSize[5] +
                d7];
        }

        public double GetValueFast(int d1, int d2, int d3, int d4, int d5, int d6, int d7, int d8)
        {
            return values[d1 * dimensionSize[0] +
                d2 * dimensionSize[1] +
                d3 * dimensionSize[2] +
                d4 * dimensionSize[3] +
                d5 * dimensionSize[4] +
                d6 * dimensionSize[5] +
                d7 * dimensionSize[6] +
                d8];
        }

        public double GetValueFast(int d1, int d2, int d3, int d4, int d5, int d6, int d7, int d8, int d9)
        {
            return values[d1 * dimensionSize[0] +
                d2 * dimensionSize[1] +
                d3 * dimensionSize[2] +
                d4 * dimensionSize[3] +
                d5 * dimensionSize[4] +
                d6 * dimensionSize[5] +
                d7 * dimensionSize[6] +
                d8 * dimensionSize[7] +
                d9];
        }

        public double GetValue()
        {
            return values[0];
        }

        public double GetValue(int d1)
        {
            if (Rank != 1)
                throw new TensorShapeException("use 1 parameter to get value, Rank must be 1!");

            return values[d1];
        }

        public double GetValue(int d1, int d2)
        {
            if (Rank != 2)
                throw new TensorShapeException("use 2 parameter to get value, Rank must be 2!");

            return values[d1 * dimensionSize[0] + d2];
        }

        public double GetValue(int d1, int d2, int d3)
        {
            if (Rank != 3)
                throw new TensorShapeException("use 3 parameter to get value, Rank must be 3!");

            return values[d1 * dimensionSize[0] + d2 * dimensionSize[1] + d3];
        }

        public double GetValue(int d1, int d2, int d3, int d4)
        {
            if (Rank != 4)
                throw new TensorShapeException("use 4 parameter to get value, Rank must be 4!");

            return values[d1 * dimensionSize[0] + d2 * dimensionSize[1] + d3 * dimensionSize[2] + d4];
        }

        public double GetValue(int d1, int d2, int d3, int d4, int d5)
        {
            if (Rank != 5)
                throw new TensorShapeException("use 5 parameter to get value, Rank must be 5!");

            return values[d1 * dimensionSize[0] + d2 * dimensionSize[1] + d3 * dimensionSize[2] + d4 * dimensionSize[3] + d5];
        }

        public double GetValue(int d1, int d2, int d3, int d4, int d5, int d6)
        {
            if (Rank != 6)
                throw new TensorShapeException("use 6 parameter to get value, Rank must be 6!");

            return values[d1 * dimensionSize[0] +
                d2 * dimensionSize[1] +
                d3 * dimensionSize[2] +
                d4 * dimensionSize[3] +
                d5 * dimensionSize[4] +
                d6];
        }

        public double GetValue(int d1, int d2, int d3, int d4, int d5, int d6, int d7)
        {
            if (Rank != 7)
                throw new TensorShapeException("use 7 parameter to get value, Rank must be 7!");

            return values[d1 * dimensionSize[0] +
                d2 * dimensionSize[1] +
                d3 * dimensionSize[2] +
                d4 * dimensionSize[3] +
                d5 * dimensionSize[4] +
                d6 * dimensionSize[5] +
                d7];
        }

        public double GetValue(int d1, int d2, int d3, int d4, int d5, int d6, int d7, int d8)
        {
            if (Rank != 8)
                throw new TensorShapeException("use 8 parameter to get value, Rank must be 8!");

            return values[d1 * dimensionSize[0] +
                d2 * dimensionSize[1] +
                d3 * dimensionSize[2] +
                d4 * dimensionSize[3] +
                d5 * dimensionSize[4] +
                d6 * dimensionSize[5] +
                d7 * dimensionSize[6] +
                d8];
        }

        public double GetValue(int d1, int d2, int d3, int d4, int d5, int d6, int d7, int d8, int d9)
        {
            if (Rank != 9)
                throw new TensorShapeException("use 9 parameter to get value, Rank must be 9!");

            return values[d1 * dimensionSize[0] +
                d2 * dimensionSize[1] +
                d3 * dimensionSize[2] +
                d4 * dimensionSize[3] +
                d5 * dimensionSize[4] +
                d6 * dimensionSize[5] +
                d7 * dimensionSize[6] +
                d8 * dimensionSize[7] +
                d9];
        }


        public void SetValueFast(double value)
        {
            values[0] = value;
        }

        public void SetValueFast(double value, int d1)
        {
            values[d1] = value;
        }

        public void SetValueFast(double value, int d1, int d2)
        {
            values[d1 * dimensionSize[0] + d2] = value;
        }

        public void SetValueFast(double value, int d1, int d2, int d3)
        {
            values[d1 * dimensionSize[0] + d2 * dimensionSize[1] + d3] = value;
        }

        public void SetValueFast(double value, int d1, int d2, int d3, int d4)
        {
            values[d1 * dimensionSize[0] + d2 * dimensionSize[1] + d3 * dimensionSize[2] + d4] = value;
        }

        public void SetValueFast(double value, int d1, int d2, int d3, int d4, int d5)
        {
            values[d1 * dimensionSize[0] + d2 * dimensionSize[1] + d3 * dimensionSize[2] + d4 * dimensionSize[3] + d5] = value;
        }

        public void SetValueFast(double value, int d1, int d2, int d3, int d4, int d5, int d6)
        {
            values[d1 * dimensionSize[0] +
                d2 * dimensionSize[1] +
                d3 * dimensionSize[2] +
                d4 * dimensionSize[3] +
                d5 * dimensionSize[4] +
                d6] = value;
        }

        public void SetValueFast(double value, int d1, int d2, int d3, int d4, int d5, int d6, int d7)
        {
            values[d1 * dimensionSize[0] +
                d2 * dimensionSize[1] +
                d3 * dimensionSize[2] +
                d4 * dimensionSize[3] +
                d5 * dimensionSize[4] +
                d6 * dimensionSize[5] +
                d7] = value;
        }

        public void SetValueFast(double value, int d1, int d2, int d3, int d4, int d5, int d6, int d7, int d8)
        {
            values[d1 * dimensionSize[0] +
                d2 * dimensionSize[1] +
                d3 * dimensionSize[2] +
                d4 * dimensionSize[3] +
                d5 * dimensionSize[4] +
                d6 * dimensionSize[5] +
                d7 * dimensionSize[6] +
                d8] = value;
        }

        public void SetValueFast(double value, int d1, int d2, int d3, int d4, int d5, int d6, int d7, int d8, int d9)
        {
            values[d1 * dimensionSize[0] +
                d2 * dimensionSize[1] +
                d3 * dimensionSize[2] +
                d4 * dimensionSize[3] +
                d5 * dimensionSize[4] +
                d6 * dimensionSize[5] +
                d7 * dimensionSize[6] +
                d8 * dimensionSize[7] +
                d9] = value;
        }


        public void SetValue(double value)
        {
            values[0] = value;
        }

        public void SetValue(double value, int d1)
        {
            if (Rank != 1)
                throw new TensorShapeException("use 1 parameter to Set value, Rank must be 1!");

            values[d1] = value;
        }

        public void SetValue(double value, int d1, int d2)
        {
            if (Rank != 2)
                throw new TensorShapeException("use 2 parameter to Set value, Rank must be 2!");

            values[d1 * dimensionSize[0] + d2] = value;
        }

        public void SetValue(double value, int d1, int d2, int d3)
        {
            if (Rank != 3)
                throw new TensorShapeException("use 3 parameter to Set value, Rank must be 3!");

            values[d1 * dimensionSize[0] + d2 * dimensionSize[1] + d3] = value;
        }

        public void SetValue(double value, int d1, int d2, int d3, int d4)
        {
            if (Rank != 4)
                throw new TensorShapeException("use 4 parameter to Set value, Rank must be 4!");

            values[d1 * dimensionSize[0] + d2 * dimensionSize[1] + d3 * dimensionSize[2] + d4] = value;
        }

        public void SetValue(double value, int d1, int d2, int d3, int d4, int d5)
        {
            if (Rank != 5)
                throw new TensorShapeException("use 5 parameter to Set value, Rank must be 5!");

            values[d1 * dimensionSize[0] + d2 * dimensionSize[1] + d3 * dimensionSize[2] + d4 * dimensionSize[3] + d5] = value;
        }

        public void SetValue(double value, int d1, int d2, int d3, int d4, int d5, int d6)
        {
            if (Rank != 6)
                throw new TensorShapeException("use 6 parameter to Set value, Rank must be 6!");

            values[d1 * dimensionSize[0] +
                d2 * dimensionSize[1] +
                d3 * dimensionSize[2] +
                d4 * dimensionSize[3] +
                d5 * dimensionSize[4] +
                d6] = value;
        }

        public void SetValue(double value, int d1, int d2, int d3, int d4, int d5, int d6, int d7)
        {
            if (Rank != 7)
                throw new TensorShapeException("use 7 parameter to Set value, Rank must be 7!");

            values[d1 * dimensionSize[0] +
                d2 * dimensionSize[1] +
                d3 * dimensionSize[2] +
                d4 * dimensionSize[3] +
                d5 * dimensionSize[4] +
                d6 * dimensionSize[5] +
                d7] = value;
        }

        public void SetValue(double value, int d1, int d2, int d3, int d4, int d5, int d6, int d7, int d8)
        {
            if (Rank != 8)
                throw new TensorShapeException("use 8 parameter to Set value, Rank must be 8!");

            values[d1 * dimensionSize[0] +
                d2 * dimensionSize[1] +
                d3 * dimensionSize[2] +
                d4 * dimensionSize[3] +
                d5 * dimensionSize[4] +
                d6 * dimensionSize[5] +
                d7 * dimensionSize[6] +
                d8] = value;
        }

        public void SetValue(double value, int d1, int d2, int d3, int d4, int d5, int d6, int d7, int d8, int d9)
        {
            if (Rank != 9)
                throw new TensorShapeException("use 9 parameter to Set value, Rank must be 9!");

            values[d1 * dimensionSize[0] +
                d2 * dimensionSize[1] +
                d3 * dimensionSize[2] +
                d4 * dimensionSize[3] +
                d5 * dimensionSize[4] +
                d6 * dimensionSize[5] +
                d7 * dimensionSize[6] +
                d8 * dimensionSize[7] +
                d9] = value;
        }


        /// <summary>
        /// 获取Tensor指定位置的值
        /// </summary>
        /// <param name="index">指定的位置，需要符合Rank</param>
        /// <returns>指定位置对应的值</returns>
        public double this[params int[] index]
        {
            get
            {
                return GetValue(index);
            }
            set
            {
                SetValue(value, index);
            }
        }

        /// <summary>
        /// 获取Tensor指定位置的值，Rank为0，也就是Tensor为标量时指定index为空返回标量的值
        /// </summary>
        /// <param name="index">指定的位置</param>
        /// <returns>指定位置返回的值</returns>
        public double GetValue(params int[] index)
        {
            if (index.Length != Rank)
                throw new TensorShapeException("index must be the same length as Rank!");

            var offset = GetRawOffset(index);
            return values[offset];
        }

        /// <summary>
        /// 将Tensor中指定位置的值设置为value，Tensor为标量时保持index为空
        /// </summary>
        /// <param name="value">要设定的值</param>
        /// <param name="index">要设定的位置</param>
        public void SetValue(double value, params int[] index)
        {
            if (index.Length != Rank)
                throw new TensorShapeException("index must be the same length as Rank!");

            var offset = GetRawOffset(index);
            values[offset] = value;
        }

        /// <summary>
        /// 获取底层数据
        /// </summary>
        /// <returns></returns>
        public double[] GetRawValues()
        {
            return values;
        }

        /// <summary>
        /// 所有元素的和
        /// </summary>
        /// <returns>和</returns>
        public double Sum()
        {
            return values.Sum();
        }

        /// <summary>
        /// 所有元素的均值
        /// </summary>
        /// <returns>均值</returns>
        public double Mean()
        {
            return values.Average();
        }

        /// <summary>
        /// 所有元素中的最大值
        /// </summary>
        /// <returns>最大值</returns>
        public double Max()
        {
            return values.Max();
        }

        /// <summary>
        /// 所有元素中的最小值
        /// </summary>
        /// <returns>最小值</returns>
        public double Min()
        {
            return values.Min();
        }

        public void Clear(double defaultValue = 0)
        {
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = 0;
            }
        }

        /// <summary>
        /// 以第一个维度为索引返回新的Tensor
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public Tensor GetTensorByDim1(int index)
        {
            if (Rank == 0)
                throw new TensorShapeException("this tensor is a scalar, don't have any dimension!");

            if (index >= shape[0])
                throw new TensorShapeException($"index out of range! index is {index} rank1 is {shape[0]}");

            if (Rank == 1)
            {
                var result = new Tensor();
                result.SetValue(this[index]);
                return result;
            }

            var len = dimensionSize[0];
            var start = index * len;
            var data = new double[len];
            var newShape = new int[Rank - 1];

            //计算新Tensor的shape
            Array.Copy(shape, 1, newShape, 0, newShape.Length);
            //复制数据到新Tensor
            Array.Copy(values, start, data, 0, len);

            return new Tensor(data, newShape);
        }

        public void GetByDim1(int index, double[] result)
        {
            var len = result.Length;
            var start = index * len;
            Array.Copy(values, start, result, 0, len);
        }
    }
}
