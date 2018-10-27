using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class NeuralNetwork : ITrain
    {
        public double Error(Vector yHat, Vector y)
        {
            throw new NotImplementedException();
        }

        public double Loss(Matrix X, Vector y)
        {
            throw new NotImplementedException();
        }

        public double Predict(Vector X)
        {
            throw new NotImplementedException();
        }

        public Vector Predict(Matrix X)
        {
            throw new NotImplementedException();
        }

        public void Step(Matrix X, Vector y)
        {
            throw new NotImplementedException();
        }
    }
}
