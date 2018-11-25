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
        public IModel Model { get; set; }
        public IPreProcessor PreProcessor { get; set; }
        public DiscreteCodec<T> LabelCodec { get; set; }
        public MachineType MachineType { get; private set; }
        public Machine(IModel model, MachineType type)
        {
            Model = model;
            MachineType = type;
        }

        public Tensor LastRawResult { get; protected set; }
        public Tensor LastResultCodec { get; protected set; }

        public List<T>Predict(Tensor X)
        {
            X = Normalize(X);
            LastRawResult = Model.Predict(X);
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
