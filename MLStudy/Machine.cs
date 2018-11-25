﻿using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;

namespace MLStudy
{
    public class Machine : Machine<double>
    {
        public Machine(IModel model, MachineType type)
            : base(model, type)
        { }

        public new List<double> Predict(Tensor X)
        {
            X = Normalize(X);
            LastRawResult = Model.Predict(X);

            if (MachineType == MachineType.Classification)
            {
                LastResultCodec = Utilities.ProbabilityToCode(LastRawResult);
                return LastResultCodec.GetRawValues().ToList();
            }
            else
                return LastRawResult.GetRawValues().ToList();
        }

        public new List<double> Predict(DataTable table)
        {
            var x = PreProcessor.PreProcess(table);
            return Predict(x);
        }
    }
}
