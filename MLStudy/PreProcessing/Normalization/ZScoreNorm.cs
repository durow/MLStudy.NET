using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.PreProcessing
{
    public class ZScoreNorm : INormalizer
    {
        public double Mean { get; private set; }
        public double Delta { get; private set; }

        public ZScoreNorm(double mean, double delta)
        {
            Mean = mean;
            Delta = delta;
        }

        public ZScoreNorm(TensorOld tensor)
        {
            Mean = tensor.Mean();
            var temp = tensor - Mean;
            temp.Apply(a => a * a);
            Delta = Math.Sqrt(temp.Mean());
        }

        public TensorOld Normalize(TensorOld input)
        {
            var result = input.GetSameShape();
            Normalize(input, result);
            return result;
        }

        public void Normalize(TensorOld input, TensorOld output)
        {
            TensorOld.Apply(input, output, a => (a - Mean) / Delta);
        }
    }
}
