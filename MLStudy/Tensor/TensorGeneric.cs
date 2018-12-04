using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public abstract class Tensor<T> where T : struct
    {
        public TensorData<T> Values { get; private set; }
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
            return Tensor.Create<T>(shape);
        }

        public abstract void AddLocal(T a);

        public abstract Tensor<T> Add(T a);

        public abstract void MinusLocal(T a);

        public abstract Tensor<T> Minus(T a);

        public abstract void MunusByLocal(T a);

        public abstract Tensor<T> MunusBy(T a);

        public static Tensor<T> operator +(Tensor<T> a, T b)
        {
            return Tensor.Add<T>(a, b);
        }
    }
}
