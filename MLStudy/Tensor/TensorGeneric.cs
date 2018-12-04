using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class TensorGeneric<T>
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

        public TensorGeneric(params int[] shape)
        {
            Values = new TensorData<T>(shape);
        }

        public TensorGeneric<T> GetTensor(params int[] index)
        {
            var data = Values.GetData(index);
            return new TensorGeneric<T>
            {
                Values = data
            };
        }

        public void SetTensor(T value, params int[] index)
        {
            Values.SetData(value, index);
        }

        public virtual TensorGeneric<T> GetSameShape()
        {
            return new TensorGeneric<T>(shape);
        }

        public virtual void AddLocal(T a)
        {
            TensorGeneric.Add(this, a, this);
        }

        public virtual TensorGeneric<T> Add(T a)
        {
            return TensorGeneric.Add(this, a);
        }

        public static TensorGeneric<T> operator +(TensorGeneric<T> a, T b)
        {
            return a.Add(b);
        }
    }
}
