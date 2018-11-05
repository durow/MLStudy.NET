using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class ShapeException: Exception
    {
        public ShapeException(string msg):base(msg)
        { }
    }

    public class TensorException: Exception
    {
        public TensorException(string msg):base(msg)
        { }
    }

    public class InputException : Exception
    {
        public InputException(string msg) : base(msg)
        { }
    }
}
