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

        public LogNorm(Tensor tensor)
        {
            Max = tensor.Max();
            Denom = Math.Log10(Max);
        }

        public Tensor Normalize(Tensor input)
        {
            var result = input.GetSameShape();
            Normalize(input, result);
            return result;
        }

        public void Normalize(Tensor input, Tensor output)
        {
            Tensor.Apply(input, output, a => Math.Log10(a) / Denom);
        }
    }
}
