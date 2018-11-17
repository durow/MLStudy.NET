using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.NN
{
    public class ReLULayer : ILayer
    {
        public Tensor LastForwardOutput { get; private set; }

        public Tensor Backward(Tensor error)
        {
            var derivative = Tensor.Apply(LastForwardOutput, Derivative);
            return derivative.MultipleElementWise(error);
        }

        public Tensor Forward(Tensor input)
        {
            LastForwardOutput = Tensor.Apply(input, Function);
            return LastForwardOutput;
        }

        public static double Function(double x)
        {
            return Math.Max(x, 0);
        }

        public static double Derivative(double x)
        {
            return x > 0 ? 1 : 0;
        }
    }
}
