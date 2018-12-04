using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;

namespace MLStudy
{
    public class ClassificationMachine : Machine
    {
        public DiscreteCodec LabelCodec { get; set; }
        public TensorOld LastCodecResult { get; protected set; }

        public ClassificationMachine(IModel model)
            : base(model)
        { }

        public List<string> Predict(TensorOld X)
        {
            X = Normalize(X);
            LastRawResult = Model.Predict(X);

            LastCodecResult = Utilities.ProbabilityToCode(LastRawResult);

            if (LabelCodec != null)
            {
                return LabelCodec.Decode(LastCodecResult);
            }
            return LastRawResult.GetRawValues().Select(a => a.ToString()).ToList();
        }

        public List<double> PredictValue(TensorOld X)
        {
            X = Normalize(X);
            LastRawResult = Model.Predict(X);

            LastCodecResult = Utilities.ProbabilityToCode(LastRawResult);
            return LastCodecResult.GetRawValues().ToList();
        }

        public List<string> Predict(DataTable table)
        {
            var x = PreProcessor.PreProcessX(table);
            return Predict(x);
        }
    }
}
