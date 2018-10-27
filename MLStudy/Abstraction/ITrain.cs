using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public interface ITrain
    {
        void Step(Matrix X, Vector y);
        double Loss(Matrix yHat, Vector y);
    }
}
