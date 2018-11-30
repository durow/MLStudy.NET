using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;

namespace MLStudy
{
    public class RegressionMachine : Machine
    {
        public RegressionMachine(IModel model)
            : base(model)
        { }

        public List<double> Predict(Tensor X)
        {
            X = Normalize(X);
            LastRawResult = Model.Predict(X);
            return LastRawResult.GetRawValues().ToList();
        }

        public List<double> Predict(DataTable table)
        {
            var x = PreProcessor.PreProcessX(table);
            return Predict(x);
        }
    }
}
