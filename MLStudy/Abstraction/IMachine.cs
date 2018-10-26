using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public interface IMachine
    {
        void Step(Matrix X, Vector y);
        double Predict(Vector X);
        Vector Predict(Matrix X);
        double Loss(Vector yHat, Vector y);
        double Error(Vector yHat, Vector y);
    }
}
