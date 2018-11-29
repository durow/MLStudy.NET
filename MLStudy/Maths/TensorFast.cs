using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Maths
{
    public sealed class TensorFast
    {
        public Array Values { get; set; }

        public int ElementCount { get { return Values.Length; } }

        public int Rank { get { return Values.Rank; } }

        internal int[] shape;

        private int[] dimensionSize; //存储各个维度的大小

        public double this[params int[] index]
        {
            get
            {
                return (double)Values.GetValue(index);
            }
            set
            {
                Values.SetValue(value, index);
            }
        }

        public TensorFast(params int[] shape)
        {
            Values = Array.CreateInstance(typeof(double), shape);
        }

        //public TensorFast(Array data)
        //{
        //    Values = data;
        //}

    }
}
