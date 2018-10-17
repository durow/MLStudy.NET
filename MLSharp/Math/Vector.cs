using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MLSharp.Math
{
    public class Vector
    {
        public readonly double[] Values;
        public double this[int index]
        {
            get
            {
                return Values[index];
            }
        }

        public Vector(double[] values)
        {
            Values = values;
        }
    }
}
