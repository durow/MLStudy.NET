using System;
using System.Collections.Generic;
using System.Text;
using System.Data;
using MLStudy.Abstraction;
using System.Linq;

namespace MLStudy
{
    public abstract class Machine
    {
        public INormalizer Normalizer { get; set; }
        public IModel Model { get; set; }
        public IPreProcessor PreProcessor { get; set; }
        public TensorOld LastRawResult { get; protected set; }

        public Machine(IModel model)
        {
            Model = model;
        }



        protected TensorOld Normalize(TensorOld input)
        {
            if (Normalizer == null)
                return input;

            return Normalizer.Normalize(input);
        }
    }
}
