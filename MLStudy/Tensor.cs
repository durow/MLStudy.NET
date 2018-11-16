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
    public class Tensor
    {
        private double[] values;
        private int[] shape;
        private int[] dimensionSize;

        public int Rank { get; private set; }
        public int[] Shape
        {
            get
            {
                return getShape();
            }
        }
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
        public long ElementCount
        {
            get
            {
                return values.Length;
            }
        }

        #region Creation

        public Tensor(Array data)
        {
            var shape = new int[data.Rank];
            for (int i = 0; i < shape.Length; i++)
            {
                shape[i] = data.GetLength(i);
            }
            InitTensor(shape);

            values = new double[data.LongLength];
            for (int i = 0; i < data.LongLength; i++)
            {
                values[i] = (double)data.GetValue(getIndex(i));
            }
        }

        public Tensor(params int[] shape)
        {
            InitTensor(shape);
            values = new double[getTotalLength(shape)];
        }

        public Tensor(double[] data, params int[] shape)
        {
            if (shape.Length == 0)
                shape = new int[] { data.Length };

            var len = getTotalLength(shape);
            if (len != data.Length)
                throw new TensorShapeException("data length and shape are not same!");

            InitTensor(shape);
            values = data;
        }

        public static Tensor Rand(params int[] shape)
        {
            if (shape.Length == 0)
                shape = new int[1];
            var len = getTotalLength(shape);
            var data = DataEmulator.Instance.RandomArray(getTotalLength(shape));
            return new Tensor(data, shape);
        }

        public static Tensor RandGaussian(params int[] shape)
        {
            if (shape.Length == 0)
                shape = new int[1];
            var len = getTotalLength(shape);
            var data = DataEmulator.Instance.RandomArrayGaussian(getTotalLength(shape));
            return new Tensor(data, shape);
        }

        public static Tensor Zeros(params int[] shape)
        {
            return new Tensor(shape);
        }

        public static Tensor Ones(params int[] shape)
        {
            if (shape.Length == 0)
                shape = new int[1];
            var len = getTotalLength(shape);
            var data = new double[len];
            for (int i = 0; i < len; i++)
            {
                data[i] = 1;
            }
            return new Tensor(data, shape);
        }

        public static Tensor Values(double value, params int[] shape)
        {
            if (shape.Length == 0)
                shape = new int[1];
            var len = getTotalLength(shape);
            var data = new double[len];
            for (int i = 0; i < len; i++)
            {
                data[i] = value;
            }
            return new Tensor(data, shape);
        }

        public static Tensor Eye(int width)
        {
            var result = new Tensor(width, width);
            for (int i = 0; i < width; i++)
            {
                result[i, i] = 1;
            }
            return result;
        }

        #endregion

        public double GetValue(params int[] index)
        {
            if (index.Length == 0 && ElementCount == 1)
                return values[0];

            if (index.Length != Rank)
                throw new TensorShapeException("index must be the same length as Rank!");

            for (int i = 0; i < Rank; i++)
            {
                if (index[i] >= shape[i])
                    throw new TensorShapeException($"index out of range! index is {index}, shape is {shape}");
            }

            if (index.Length == 0)
                return values[0];

            var offset = getOffset(index);
            return values[offset];
        }

        public void SetValue(double value, params int[] index)
        {
            if (index.Length == 0 && ElementCount == 1)
            {
                values[0] = value;
                return;
            }

            if (index.Length != Rank)
                throw new TensorShapeException("index must be the same length as Rank!");

            for (int i = 0; i < Rank; i++)
            {
                if (index[i] >= shape[i])
                    throw new TensorShapeException($"index out of range! index is {index}, shape is {shape}");
            }

            if (index.Length == 0)
            {
                values[0] = value;
                return;
            }

            var offset = getOffset(index);
            values[offset] = value;
        }

        public Tensor Reshape(params int[] shape)
        {
            if (shape.Length == 0)
                return new Tensor(values);

            var len = getTotalLength(shape);
            if (len != ElementCount)
                throw new TensorShapeException($"can't reshape {this.shape.ToString()} to {shape.ToString()}");

            return new Tensor(values, shape);
        }

        public Tensor Transpose()
        {
            var result = new Tensor(shape.Reverse().ToArray());
            for (int i = 0; i < ElementCount; i++)
            {
                var index = getIndex(i);
                var transIndex = index.Reverse().ToArray();
                result[transIndex] = values[i];
            }
            return result;
        }

        public Tensor GetSameShape()
        {
            return new Tensor(shape);
        }

        public Tensor Clone()
        {
            var result = GetSameShape();
            values.CopyTo(result.values, 0);
            return result;
        }

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

        public static Tensor Apply(Tensor tensor, Func<double, double> function)
        {
            return tensor.Clone().Apply(function);
        }

        #region Add

        public Tensor Add(double d)
        {
            Apply(a => a + d);
            return this;
        }

        public Tensor Add(Tensor t)
        {
            if (t.ElementCount == 1)
                return Add(t.GetValue());

            CheckShape(shape, t.shape);

            Parallel.ForEach(Partitioner.Create(0, values.Length),
                arg =>
                {
                    for (long i = arg.Item1; i < arg.Item2; i++)
                    {
                        values[i] += t.values[i];
                    }
                });
            return this;
        }

        public static Tensor Add(Tensor t, double d)
        {
            return t.Clone().Add(d);
        }

        public static Tensor Add(Tensor a, Tensor b)
        {
            if (a.ElementCount == 1)
                return Add(b, a.GetValue());
            if (b.ElementCount == 1)
                return Add(a, b.GetValue());

            return a.Clone().Add(b);
        }

        public static Tensor operator +(Tensor t, double d)
        {
            return Add(t, d);
        }

        public static Tensor operator +(double d, Tensor t)
        {
            return Add(t, d);
        }

        public static Tensor operator +(Tensor a, Tensor b)
        {
            return Add(a, b);
        }

        #endregion

        #region Minus

        public Tensor Minus(double d)
        {
            Apply(a => a - d);
            return this;
        }

        public Tensor MinusBy(double d)
        {
            Apply(a => d - a);
            return this;
        }

        public Tensor Minus(Tensor t)
        {
            if (t.ElementCount == 1)
                return Minus(t.GetValue());

            CheckShape(shape, t.shape);

            Parallel.ForEach(Partitioner.Create(0, values.Length),
                arg =>
                {
                    for (long i = arg.Item1; i < arg.Item2; i++)
                    {
                        values[i] -= t.values[i];
                    }
                });
            return this;
        }

        public static Tensor Minus(Tensor t, double d)
        {
            return t.Clone().Minus(d);
        }

        public static Tensor Minus(double d, Tensor t)
        {
            return t.Clone().MinusBy(d);
        }

        public static Tensor Minus(Tensor a, Tensor b)
        {
            if (a.ElementCount == 1)
                return Minus(a.GetValue(), b);
            if (b.ElementCount == 1)
                return Minus(a, b.GetValue());

            return a.Clone().Minus(b);
        }

        public static Tensor operator -(Tensor t, double d)
        {
            return Minus(t, d);
        }

        public static Tensor operator -(double d, Tensor t)
        {
            return Minus(d, t);
        }

        public static Tensor operator -(Tensor a, Tensor b)
        {
            return Minus(a, b);
        }

        #endregion

        #region Multiple

        public Tensor Multiple(double d)
        {
            Apply(a => a * d);
            return this;
        }

        public Tensor MultipleElementWise(Tensor t)
        {
            if (t.ElementCount == 1)
                return Multiple(t.GetValue());

            CheckShape(shape, t.shape);

            Parallel.ForEach(Partitioner.Create(0, values.Length),
                arg =>
                {
                    for (long i = arg.Item1; i < arg.Item2; i++)
                    {
                        values[i] *= t.values[i];
                    }
                });
            return this;
        }

        public static Tensor Multiple(Tensor t, double d)
        {
            return t.Clone().Multiple(d);
        }

        public static Tensor Multiple(Tensor a, Tensor b)
        {
            if (a.ElementCount == 1)
                return Multiple(b, a.GetValue());
            if (b.ElementCount == 1)
                return Multiple(a, b.GetValue());

            if (a.Rank == 1)
                a = a.Reshape(1, (int)a.ElementCount);
            if (b.Rank == 1)
                b = b.Reshape((int)b.ElementCount, 1);

            CheckMultipleShape(a, b);
            var result = new Tensor(a.shape[0], b.shape[1]);

            Parallel.For(0, a.shape[0], i =>
            {
                Parallel.For(0, b.shape[1], j =>
                {
                    var sum = 0d;
                    for (int k = 0; k < a.shape[1]; k++)
                    {
                        sum += a[i, k] * b[k, j];
                    }
                    result[i, j] = sum;
                });
            });

            return result;
        }

        public static Tensor MultipleElementWise(Tensor a, Tensor b)
        {
            if (a.ElementCount == 1)
                return Multiple(b, a.GetValue());
            if (b.ElementCount == 1)
                return Multiple(a, b.GetValue());

            return a.Clone().MultipleElementWise(b);
        }

        public static Tensor operator *(Tensor t, double d)
        {
            return Multiple(t, d);
        }

        public static Tensor operator *(double d, Tensor t)
        {
            return Multiple(t, d);
        }

        public static Tensor operator *(Tensor a, Tensor b)
        {
            return Multiple(a, b);
        }

        #endregion

        #region Divide

        public Tensor Divide(double d)
        {
            Apply(a => a / d);
            return this;
        }

        public Tensor DivideBy(double d)
        {
            Apply(a => d / a);
            return this;
        }

        public Tensor DivideElementWise(Tensor t)
        {
            if (t.ElementCount == 1)
                return Divide(t.GetValue());

            CheckShape(shape, t.shape);

            Parallel.ForEach(Partitioner.Create(0, values.Length),
                arg =>
                {
                    for (long i = arg.Item1; i < arg.Item2; i++)
                    {
                        values[i] /= t.values[i];
                    }
                });
            return this;
        }

        public static Tensor Divide(Tensor t, double d)
        {
            return t.Clone().Divide(d);
        }

        public static Tensor Divide(double d, Tensor t)
        {
            return t.Clone().DivideBy(d);
        }

        public static Tensor DivideElementWise(Tensor a, Tensor b)
        {
            if (a.ElementCount == 1)
                return Divide(b, a.GetValue());
            if (b.ElementCount == 1)
                return Divide(a, b.GetValue());

            return a.Clone().DivideElementWise(b);
        }

        public static Tensor operator /(Tensor t, double d)
        {
            return Divide(t, d);
        }

        public static Tensor operator /(double d, Tensor t)
        {
            return Divide(d, t);
        }

        #endregion

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

        public Tensor GetTensorByDim1(int index)
        {
            if (index >= shape[0])
                throw new TensorShapeException($"index out of range! index is {index} rank1 is {shape[0]}");

            var len = dimensionSize[0];
            var start = index * len;
            var data = new double[len];
            var newShape = new int[Rank - 1];
            for (int i = 0; i < newShape.Length; i++)
            {
                newShape[i] = shape[i + 1];
            }

            for (int i = 0; i < len; i++)
            {
                data[i] = values[start + i];
            }

            return new Tensor(data, newShape);
        }

        private void InitTensor(int[] shape)
        {
            if (shape.Length == 0)
            {
                shape = new int[1];
                Rank = 0;
            }
            else
            {
                Rank = shape.Length;
            }

            CheckShape(shape);
            this.shape = shape;
            setDimensionSize();
        }

        private int[] getShape()
        {
            var result = new int[shape.Length];
            shape.CopyTo(result, 0);
            return result;
        }

        private static int getTotalLength(int[] shape)
        {
            var result = 1;
            for (int i = 0; i < shape.Length; i++)
            {
                result *= shape[i];
            }
            return result;
        }

        private void setDimensionSize()
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

        private int getOffset(int[] index)
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

        private int[] getIndex(int offset)
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

        private static void CheckShape(int[] shape)
        {
            if (shape.Length == 0)
                throw new TensorShapeException("Tensor rank must > 0 !");
            foreach (var item in shape)
            {
                if (item <= 0)
                    throw new TensorShapeException("Tensor shape must > 0 !");
            }
        }

        private static void CheckShape(int[] shape1, int[] shape2)
        {
            if (shape1.Length == shape2.Length)
                throw new TensorShapeException("Tensor rank must > 0 !");

            for (int i = 0; i < shape1.Length; i++)
            {
                if (shape1[i] != shape2[i])
                    throw new TensorShapeException("shape are not the same!");
            }
        }

        private static void CheckMultipleShape(Tensor a, Tensor b)
        {
            if (a.Rank != 2 || b.Rank != 2)
                throw new NotImplementedException("only suport multiple between scalar, vector and matrix!");

            if (a.shape[1] != b.shape[0])
                throw new TensorShapeException($"can't multiple matrix between {a.shape.ToString()} and {b.shape.ToString()}");
        }
    }
}
