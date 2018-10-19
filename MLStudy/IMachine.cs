using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public interface IMachine
    {
        object Step(Matrix X, Vector y);
        Vector Predict(Matrix X);
    }
}
