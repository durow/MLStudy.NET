using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public abstract class TensorA
    {
        public T this[int index]
        {
            get
            {
                return GetValue<T>(index);
            }
        }

        public abstract T GetValue<T>(params int[] index);
    }
}
