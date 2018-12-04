using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.PreProcessing
{
    public class LogNorm : INormalizer
    {
        public double Max { get; private set; }
        public double Denom { get; private set; }

        public LogNorm(TensorOld tensor)
        {
            Max = tensor.Max();
            Denom = Math.Log10(Max);
        }

        public TensorOld Normalize(TensorOld input)
        {
            var result = input.GetSameShape();
            Normalize(input, result);
            return result;
        }

        public void Normalize(TensorOld input, TensorOld output)
        {
            TensorOld.Apply(input, output, a => Math.Log10(a) / Denom);
        }
    }
}
