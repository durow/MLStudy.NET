using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MLSharp.Math
{
    public class Vector<T>
    {
        private T[] data;
        public VectorDirection Direction { get; private set; } = VectorDirection.Column;
        public int Dimension { get { return data == null ? 0 : data.Length; } }

        public Vector(IEnumerable<T> data)
        {
            this.data = data.ToArray();
        }

        public Vector(params T[] data)
        {
            this.data = data;
        }

        public static Vector<T> operator +(Vector<T> a, Vector<T> b)
        {
            if (a.Dimension != b.Dimension)
                throw new VectorException("向量维数不同，无法相加!");

            return new Vector<T>();
        }
    }

    public enum VectorDirection
    {
        Row,
        Column
    }

    public class VectorException:Exception
    {
        public VectorException(string msg):base(msg)
        {
        }
    }
}
