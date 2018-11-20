using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public sealed class GradientDescent : IOptimizer
    {
        public double LearningRate { get; set; }

        public GradientDescent(double learningRate)
        {
            LearningRate = learningRate;
        }

        public void Optimize(Tensor target, Tensor graident)
        {
            
        }
    }
}
