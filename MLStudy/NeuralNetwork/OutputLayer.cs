using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class OutputLayer
    {
        public double LearningRate { get; set; }

        public Matrix Forward(Matrix input)
        {
            return new Matrix();
        }

        public Matrix Backward(Vector y)
        {
            return new Matrix();
        }

        public Vector Predict(Matrix X)
        {
            return new Vector();
        }
    }
}
