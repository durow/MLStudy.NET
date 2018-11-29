/*
 * Description:张量相关操作，实现了深度学习中用到的大多数计算，加减乘除和转置等。
 *             通过在一维数组的基础上加了一层中间件，用来映射张量的结构和相关操作。
 *             部分张量计算使用了Parallel进行了并行化。
 * Author:Yunxiao An
 * Date:2018.11.16
 */

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MLStudy
{
    /// <summary>
    /// 张量
    /// </summary>
    public sealed partial class Tensor
    {
        internal double[] values; //存放底层数据
        internal int[] shape; //Tensor结构信息
        private int[] dimensionSize; //存储各个维度的大小

        /// <summary>
        /// Tensor的阶
        /// </summary>
        public int Rank { get { return shape.Length; } }

        /// <summary>
        /// Tensor的结构
        /// </summary>
        public int[] Shape
        {
            get
            {
                return GetShape();
            }
        }

        /// <summary>
        /// Tensor中所有元素的个数
        /// </summary>
        public int ElementCount
        {
            get
            {
                return values.Length;
            }
        }

        /// <summary>
        /// 从data数组创建同样结构的Tensor
        /// </summary>
        /// <param name="data">数组，需要类型为double</param>
        public Tensor(Array data)
        {
            var shape = new int[data.Rank];
            for (int i = 0; i < shape.Length; i++)
            {
                shape[i] = data.GetLength(i);
            }
            InitTensorShapeInfo(shape);

            values = new double[data.LongLength];
            for (int i = 0; i < data.LongLength; i++)
            {
                values[i] = (double)data.GetValue(GetIndex(i));
            }
        }

        /// <summary>
        /// 创建一个由shape指定结构的Tensor，shape为空则创建Rank为0的标量
        /// </summary>
        /// <param name="shape">Tensor的结构</param>
        public Tensor(params int[] shape)
        {
            InitTensorShapeInfo(shape);
            values = new double[GetTotalLength(shape)];
        }

        /// <summary>
        /// 使用一维数组创建Tensor并转为shape指定的结构
        /// </summary>
        /// <param name="data">一维数组</param>
        /// <param name="shape">Tensor的结构，省略则按照一维Tensor(也就是Vector)处理</param>
        public Tensor(double[] data, params int[] shape)
        {
            if (shape.Length == 0)
                shape = new int[] { data.Length };

            var len = GetTotalLength(shape);
            if (len != data.Length)
                throw new TensorShapeException("data length and shape are not same!");

            InitTensorShapeInfo(shape);
            values = data;
        }

        /// <summary>
        /// 转换Tensor的结构，新结构与原结构需要在元素数量上一致。
        /// 新的Tensor是原Tensor的新视图，与原Tensor共享同一个底层数据。
        /// </summary>
        /// <param name="shape">要转换的新的结构</param>
        /// <returns>转换后的Tensor</returns>
        public Tensor Reshape(params int[] shape)
        {
            if (shape.Length == 0)
                return new Tensor(values);

            var len = GetTotalLength(shape);
            if (len != ElementCount)
                throw new TensorShapeException($"can't reshape {this.shape.ToString()} to {shape.ToString()}");

            return new Tensor(values, shape);
        }

        /// <summary>
        /// 把当前Tensor转置，转置后返回的是一个新的Tensor
        /// </summary>
        /// <returns>转置后的Tensor</returns>
        public Tensor Transpose()
        {
            var result = new Tensor(shape.Reverse().ToArray());
            for (int i = 0; i < ElementCount; i++)
            {
                var index = GetIndex(i);
                var transIndex = index.Reverse().ToArray();
                result[transIndex] = values[i];
            }
            return result;
        }

        public int GetRawOffset()
        {
            return 0;
        }

        public int GetRawOffset(int index)
        {
            return index;
        }

        public int GetRawOffset(int d1, int d2)
        {
            return d1 * dimensionSize[0] + d2;
        }

        public int GetRawOffset(int d1, int d2, int d3)
        {
            return d1 * dimensionSize[0] + d2 * dimensionSize[1] + d3;
        }

        public int GetRawOffset(int d1, int d2, int d3, int d4)
        {
            return d1 * dimensionSize[0] + d2 * dimensionSize[1] + d3 * dimensionSize[2] + d4;
        }

        public int GetRawOffset(int d1, int d2, int d3, int d4, int d5)
        {
            return d1 * dimensionSize[0] + d2 * dimensionSize[1] + d3 * dimensionSize[2] + d4 * dimensionSize[3] + d5;
        }

        public int GetRawOffset(params int[] index)
        {
            if (index.Length == 1)
                return index[0];

            var result = 0;
            for (int i = 0; i < dimensionSize.Length; i++)
            {
                result += dimensionSize[i] * index[i];
            }
            result += index[index.Length - 1];

            return result;
        }

        /// <summary>
        /// 创建一个新的Tensor并与现有Tensor结构相同
        /// </summary>
        /// <returns>新的Tensor</returns>
        public Tensor GetSameShape()
        {
            return new Tensor(shape);
        }

        /// <summary>
        /// 复制当前Tensor，包括结构和数据
        /// </summary>
        /// <returns></returns>
        public Tensor Clone()
        {
            var result = GetSameShape();
            Array.Copy(values, 0, result.values, 0, ElementCount);
            return result;
        }

        /// <summary>
        /// 当前Tensor的所有元素应用function，结果保存在当前Tensor
        /// </summary>
        /// <param name="function">要应用的function</param>
        /// <returns>当前Tensor</returns>
        public Tensor Apply(Func<double, double> function)
        {
            Parallel.ForEach(Partitioner.Create(0, values.Length),
                arg =>
                {
                    for (long i = arg.Item1; i < arg.Item2; i++)
                    {
                        values[i] = function(values[i]);
                    }
                });
            return this;
        }

        public Tensor Apply(Tensor a, Func<double, double, double> function)
        {
            Parallel.ForEach(Partitioner.Create(0, values.Length),
                arg =>
                {
                    for (long i = arg.Item1; i < arg.Item2; i++)
                    {
                        values[i] = function(values[i], a.values[i]);
                    }
                });
            return this;
        }

        /// <summary>
        /// 指定Tensor的每个元素应用function，结果返回为新的Tensor
        /// </summary>
        /// <param name="tensor">应用function的Tensor</param>
        /// <param name="function">引用的function</param>
        /// <returns>结果返回为新的Tensor</returns>
        public static Tensor Apply(Tensor tensor, Func<double, double> function)
        {
            var result = tensor.GetSameShape();
            Apply(tensor, result, function);
            return result;
        }

        public static Tensor Apply(Tensor a, Tensor b, Func<double, double, double> function)
        {
            var result = a.GetSameShape();
            Apply(a, b, result, function);
            return result;
        }

        /// <summary>
        /// 把input中的每个元素应用function，并将结果写入到result中。
        /// 必要的时候在调用这个方法前进行Tensor结构一致性检查
        /// </summary>
        /// <param name="input">输入的Tensor</param>
        /// <param name="result">写入结果的Tensor</param>
        /// <param name="function">应用的运算</param>
        public static void Apply(Tensor input, Tensor result, Func<double, double> function)
        {
            //这个方法中不进行Tensor结构一致性的检查
            //所有的Tensor结构的问题都放到Prepare过程中
            //或者必要的时候在调用这个函数之前执行Tensor结构一致性的检查

            Parallel.ForEach(Partitioner.Create(0, input.values.Length),
                arg =>
                {
                    for (long i = arg.Item1; i < arg.Item2; i++)
                    {
                        result.values[i] = function(input.values[i]);
                    }
                });
        }

        /// <summary>
        /// a和b对应位置元素执行function操作，结果写入result对应位置，要求a，b，result结构一致
        /// 该方法不做结构一致性检查，必要时在调用之前检查。
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="result"></param>
        /// <param name="function"></param>
        public static void Apply(Tensor a, Tensor b, Tensor result, Func<double, double, double> function)
        {
            //这个方法中不进行Tensor结构一致性的检查
            //所有的Tensor结构的问题都放到Prepare过程中
            //或者必要的时候在调用这个函数之前执行Tensor结构一致性的检查

            Parallel.ForEach(Partitioner.Create(0, result.values.Length),
                arg =>
                {
                    for (long i = arg.Item1; i < arg.Item2; i++)
                    {
                        result.values[i] = function(a.values[i], b.values[i]);
                    }
                });
        }

        #region Override

        public override bool Equals(object o)
        {
            if (!(o is Tensor tensor))
                return false;

            if (Rank != tensor.Rank)
                return false;

            for (int i = 0; i < Rank; i++)
            {
                if (shape[i] != shape[i])
                    return false;
            }

            for (int i = 0; i < values.Length; i++)
            {
                if (values[i] != tensor.values[i])
                    return false;
            }

            return true;
        }

        public override string ToString()
        {
            if (Rank == 1)
            {
                var content = string.Join(", ", values);
                return $"[{content}]";
            }
            else
            {
                var result = new List<string>();
                for (int i = 0; i < shape[0]; i++)
                {
                    result.Add(GetTensorByDim1(i).ToString());
                }
                var content = string.Join(",\n", result);
                return $"[{content}]";
            }
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        #endregion

        #region Helper Methods

        private void InitTensorShapeInfo(int[] shape)
        {
            CheckShape(shape);
            this.shape = shape;
            SetDimensionSize();
        }

        private int[] GetShape()
        {
            var result = new int[shape.Length];
            //复制shape，防止被修改
            Array.Copy(shape, 0, result, 0, shape.Length);
            return result;
        }

        private static int GetTotalLength(int[] shape)
        {
            if (shape.Length == 0)
                return 1;

            var result = 1;
            for (int i = 0; i < shape.Length; i++)
            {
                result *= shape[i];
            }
            return result;
        }

        private void SetDimensionSize()
        {
            if (Rank <= 1)
                return;

            dimensionSize = new int[Rank - 1];
            for (int i = 0; i < dimensionSize.Length; i++)
            {
                var temp = 1;
                for (int j = i + 1; j < shape.Length; j++)
                {
                    temp *= shape[j];
                }
                dimensionSize[i] = temp;
            }
        }

        private void CorrectIndex(int[] index)
        {
            for (int i = 0; i < index.Length; i++)
            {
                while (index[i] < 0)
                {
                    index[i] += shape[i];
                }
            }
        }

        private int[] GetIndex(int offset)
        {
            var rest = offset;
            var result = new int[Rank];
            for (int i = 0; i < dimensionSize.Length; i++)
            {
                result[i] = rest / dimensionSize[i];
                rest = rest % dimensionSize[i];
            }
            result[Rank - 1] = rest;

            return result;
        }

        public static void CheckShape(int[] shape)
        {
            foreach (var item in shape)
            {
                if (item <= 0)
                    throw new TensorShapeException("Tensor shape must > 0 !");
            }
        }

        public static void CheckShape(int[] shape1, int[] shape2)
        {
            if (shape1.Length != shape2.Length)
                throw new TensorShapeException("Tensor rank are not same!");

            for (int i = 0; i < shape1.Length; i++)
            {
                if (shape1[i] != shape2[i])
                    throw new TensorShapeException("shape are not the same!");
            }
        }

        public static void CheckShape(Tensor t1, Tensor t2)
        {
            CheckShape(t1.shape, t2.shape);
        }

        public static bool CheckShapeBool(Tensor t1, Tensor t2)
        {
            if (t1.shape.Length != t2.shape.Length)
                return false;

            for (int i = 0; i < t1.shape.Length; i++)
            {
                if (t1.shape[i] != t2.shape[i])
                    return false;
            }
            return true;
        }

        public static void CheckMultipleShape(Tensor a, Tensor b)
        {
            if (a.Rank != 2 || b.Rank != 2)
                throw new NotImplementedException("only suport multiple between scalar, vector and matrix!");

            if (a.shape[1] != b.shape[0])
                throw new TensorShapeException($"can't multiple matrix between {a.shape.ToString()} and {b.shape.ToString()}");
        }

        #endregion
    }
}
