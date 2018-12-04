using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public abstract class Tensor<T> where T : struct
    {
        public TensorData<T> Values { get; protected set; }
        public int Rank { get { return Values.shape.Length; } }
        internal int[] shape { get { return Values.shape; } }
        public int[] Shape
        {
            get
            {
                var result = new int[Rank];
                Array.Copy(shape, 0, result, 0, Rank);
                return result;
            }
        }
        public int Count { get { return Values.Count; } }
        public Tensor<T> this[params int[] index]
        {
            get
            {
                return GetTensor(index);
            }
        }

        public Tensor(params int[] shape)
        {
            Values = new TensorData<T>(shape);
        }

        public Tensor(TensorData<T> data)
        {
            Values = data;
        }

        public abstract Tensor<T> GetTensor(params int[] index);

        public void SetTensor(T value, params int[] index)
        {
            Values.SetData(value, index);
        }

        public void SetValue(T value)
        {

        }

        public virtual Tensor<T> GetSameShape()
        {
            return Tensor.Empty<T>(shape);
        }

        public override string ToString()
        {
            if (Rank == 1)
            {
                var content = string.Join(", ", Values.GetValues());
                return $"[{content}]";
            }
            else
            {
                var result = new List<string>();
                for (int i = 0; i < shape[0]; i++)
                {
                    result.Add(this[i].ToString());
                }
                string content;
                if(Rank == 2)
                    content = string.Join(",\n", result);
                else
                    content = string.Join(",\n\n", result);
                return $"[{content}]";
            }
        }

        public abstract Tensor<T> ReShape(params int[] index);

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
