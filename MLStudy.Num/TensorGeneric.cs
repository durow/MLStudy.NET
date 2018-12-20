using System;
using System.Collections.Generic;

namespace MLStudy.Num
{
    public abstract class Tensor<T> where T : struct
    {
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

        public Tensor<T> GetSubTensor(int[] index)
        {
            var subData = Values.GetSubData(index);
            return Tensor.Create(subData);
        }

        public Tensor<T> ReShape(params int[] newShape)
        {
            var newData = Values.ReShape(newShape);
            return Tensor.Create(newData);
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
    }
}
