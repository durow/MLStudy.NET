using System;
using System.Collections.Generic;

namespace MLStudy.Num
{
    public abstract class Tensor<T> where T : struct
    {
        public Type DataType { get;}
        public TensorData<T> Values { get; }
        public Span<T> RawValues => Values.RawValues;
        public TensorShape Shape => Values.shape;
        public int Rank => Values.shape.Rank;
        public int Length => Values.shape.TotalLength;
        public Tensor<T> this[params int[] index]
        {
            get
            {
                return GetSubTensor(index);
            }
        }

        public Tensor(TensorData<T> data)
        {
            Values = data;
            DataType = typeof(T);
        }

        public void Fill(T value)
        {
            Values.RawValues.Fill(value);
        }

        public Span<T> GetSpan(params int[] index)
        {
            return Values.GetSubValues(index);
        }

        public Tensor<T> CreateSameShape()
        {
            return Tensor.Empty<T>(Values.shape.shape);
        }

        public Tensor<T> View(params int[] newShape)
        {
            return ReShape(newShape);
        }

        public override string ToString()
        {
            if (Rank == 1)
            {
                var content = string.Join(", ", Values.RawValues.ToArray());
                return $"[{content}]";
            }
            else
            {
                var result = new List<string>();
                var dim1 = Values.shape.shape[0];
                for (int i = 0; i < dim1; i++)
                {
                    result.Add(this[i].ToString());
                }
                string content;
                if (Rank == 2)
                    content = string.Join(",\n", result);
                else
                    content = string.Join(",\n\n", result);
                return $"[{content}]";
            }
        }

        public abstract Tensor<T> ReShape(params int[] newShape);
        public abstract Tensor<T> GetSubTensor(int[] index);

        public abstract void AddLocal(T a);

        public abstract Tensor<T> Add(T a);

        public abstract void AddLocal(Tensor<T> a);

        public abstract Tensor<T> Add(Tensor<T> a);

        public abstract void MinusLocal(T a);

        public abstract Tensor<T> Minus(T a);

        public abstract void MunusByLocal(T a);

        public abstract Tensor<T> MinusBy(T a);

        public abstract void MinusLocal(Tensor<T> a);

        public abstract Tensor<T> Minus(Tensor<T> a);

        public abstract void MultipleLocal(T a);

        public abstract Tensor<T> Multiple(T a);

        public abstract Tensor<T> Multiple(Tensor<T> a);

        public abstract void MultipleElementWiseLocal(Tensor<T> a);

        public abstract Tensor<T> MultipleElementWise(Tensor<T> a);

        public abstract void DivideLocal(T a);

        public abstract Tensor<T> Divide(T a);

        public abstract void DivideByLocal(T a);

        public abstract Tensor<T> DivideBy(T a);

        public abstract void DivideElementWiseLocal(Tensor<T> a);

        public abstract Tensor<T> DivideElementWise(Tensor<T> a);

        public static Tensor<T> operator +(Tensor<T> a, T b)
        {
            return a.Add(b);
        }

        public static Tensor<T> operator +(T b, Tensor<T> a)
        {
            return a.Add(b);
        }

        public static Tensor<T> operator +(Tensor<T> left, Tensor<T> right)
        {
            return left.Add(right);
        }

        public static Tensor<T> operator -(Tensor<T> a, T b)
        {
            return a.Minus(b);
        }

        public static Tensor<T> operator -(T b, Tensor<T> a)
        {
            return a.MinusBy(b);
        }

        public static Tensor<T> operator -(Tensor<T> left, Tensor<T> right)
        {
            return left.Minus(right);
        }

        public static Tensor<T> operator *(Tensor<T> a, T b)
        {
            return a.Multiple(b);
        }

        public static Tensor<T> operator *(T b, Tensor<T> a)
        {
            return a.Multiple(b);
        }

        public static Tensor<T> operator *(Tensor<T> left, Tensor<T> right)
        {
            return left.Multiple(right);
        }

        public static Tensor<T> operator /(Tensor<T> a, T b)
        {
            return a.Divide(b);
        }

        public static Tensor<T> operator /(T b, Tensor<T> a)
        {
            return a.DivideBy(b);
        }
    }
}
