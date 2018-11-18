using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Deep
{
    public class Sigmoid : ILayer
    {
        public string Name { get; set; }
        public Tensor LastForwardOutput { get; private set; }

        public Tensor Backward(Tensor error)
        {
            return Tensor
                .Apply(LastForwardOutput, DerivativeFromOutput)
                .MultipleElementWise(error);
        }

        public Tensor Forward(Tensor input)
        {
            LastForwardOutput = Tensor.Apply(input, Function);
            return LastForwardOutput;
        }

        public static double Function(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        public static double Derivative(double x)
        {
            var output = Function(x);
            return DerivativeFromOutput(output);
        }

        public static double DerivativeFromOutput(double output)
        {
            return output * (1 - output);
        }
    }
}
