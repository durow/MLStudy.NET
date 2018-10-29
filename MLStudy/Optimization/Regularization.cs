using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class Regularization
    {
        public double RegularizationWeight { get; set; } = 0.1;
        public RegularTypes RegularType { get; set; } = RegularTypes.None;

        public double GetValue(double weight)
        {
            if (RegularType == RegularTypes.L1)
                return RegularizationWeight;

            if (RegularType == RegularTypes.L2)
                return weight * RegularizationWeight;

            return 0;
        }

        public Vector GetValue(Vector weights)
        {
            if (RegularType == RegularTypes.L1)
                return new Vector(weights.Length, RegularizationWeight);

            if (RegularType == RegularTypes.L2)
                return weights * RegularizationWeight;

            return new Vector(weights.Length);
        }

        public Matrix GetValue(Matrix weights)
        {
            if (RegularType == RegularTypes.L1)
                return new Matrix(weights.Rows, weights.Columns, RegularizationWeight);

            if (RegularType == RegularTypes.L2)
                return weights * RegularizationWeight;

            return new Matrix(weights.Rows, weights.Columns);
        }
    }

    public enum RegularTypes
    {
        None,
        L1,
        L2
    }
}
