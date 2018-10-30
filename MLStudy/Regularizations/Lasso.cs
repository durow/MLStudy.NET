using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Regularizations
{
    public class Lasso : WeightDecay
    {
        public string Name => "Lasso";

        public override double GetValue(double weight)
        {
            return Weight;
        }

        public override Vector GetValue(Vector weights)
        {
            return new Vector(weights.Length, Weight);
        }

        public override Matrix GetValue(Matrix weights)
        {
            return new Matrix(weights.Rows, weights.Columns, Weight);
        }
    }
}
