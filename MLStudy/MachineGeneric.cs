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
        public MachineType MachineType { get; private set; }
        public Machine(IEngine engine, MachineType type)
        {
            Engine = engine;
            MachineType = type;
        }

        public Tensor LastRawResult { get; protected set; }
        public Tensor LastResultCodec { get; protected set; }

        public List<T>Predict(Tensor X)
        {
            X = Normalize(X);
            LastRawResult = Engine.Predict(X);
            if (LabelCodec == null)
            {
                throw new Exception("you need a LabelDecoder!");
            }

            if (MachineType == MachineType.Classification)
            {
                LastResultCodec = Utilities.ProbabilityToCode(LastRawResult);
            }

            return LabelCodec.Decode(LastResultCodec);
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

    public enum MachineType
    {
        Regression,
        Classification
    }
}
