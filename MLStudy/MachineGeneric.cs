using System;
using System.Collections.Generic;
using System.Text;
using System.Data;
using MLStudy.Abstraction;
using System.Linq;

namespace MLStudy
{
    public class Machine<T>
    {
        public INormalizer Normalizer { get; set; }
        public IEngine Engine { get; set; }
        public IPreProcessor PreProcessor { get; set; }
        public DiscreteCodec<T> LabelCodec { get; set; }

        public Machine(IEngine engine)
        {
            Engine = engine;
        }

        public List<T> Predict(Tensor X)
        {
            X = Normalize(X);
            var result = Engine.Predict(X);
            if(LabelCodec == null)
            {
                throw new Exception("you need a LabelDecoder!");
            }

            return LabelCodec.Decode(result);
        }

        public List<T> Predict(DataTable table)
        {
            var x = PreProcessor.PreProcess(table);
            return Predict(x);
        }

        protected Tensor Normalize(Tensor input)
        {
            if (Normalizer == null)
                return input;

            return Normalizer.Normalize(input);
        }
    }
}
