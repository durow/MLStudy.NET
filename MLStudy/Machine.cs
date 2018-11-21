using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;

namespace MLStudy
{
    public class Machine : Machine<double>
    {
        public Machine(IEngine engine)
            : base(engine)
        { }

        public new List<double> Predict(Tensor X)
        {
            X = Normalize(X);
            var result = Engine.Predict(X);
            return result.GetRawValues().ToList();
        }

        public new List<double> Predict(DataTable table)
        {
            var x = PreProcessor.PreProcess(table);
            return Predict(x);
        }
    }
}
