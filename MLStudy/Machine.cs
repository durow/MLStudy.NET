using System;
using System.Collections.Generic;
using System.Text;
using System.Data;
using MLStudy.Abstraction;

namespace MLStudy
{
    public class Machine
    {
        public INormalizer Normalizer { get; set; }
        public IEngine Engine { get; set; }

        Tensor X;
        Tensor y;

        public Machine(IEngine engine)
        {

        }

        public void Train()
        {

        }
    }
}
