using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class TensorShapeException: Exception
    {
        public TensorShapeException(string msg):base(msg)
        { }
    }

    public class InputException : Exception
    {
        public InputException(string msg) : base(msg)
        { }
    }
}
