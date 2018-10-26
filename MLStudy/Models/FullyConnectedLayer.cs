using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class FullyConnectedLayer
    {
        public Matrix Weights { get; private set; }
        public Vector Bias { get; private set; }

        public Vector this[int index]
        {
            get
            {
                return Weights.GetColumn(index);
            }
        }

        public void Backward()
        {
            throw new NotImplementedException();
        }

        public Matrix Forward(Matrix X)
        {
            return X * Weights;
        }
    }
}
