using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Deep
{
    public class Tanh : ILayer
    {
        public string Name { get; set; }
        public Tensor LastForwardOutput { get; private set; }

        public Tensor Backward(Tensor error)
        {
            return Tensor.Apply(LastForwardOutput, DerivativeByOutput)
                .MultipleElementWise(error);
        }

        public Tensor Forward(Tensor input)
        {
            LastForwardOutput = Tensor.Apply(input, Function);
            return LastForwardOutput;
        }

        public static double Function(double x)
        {
            var pos = Math.Exp(x);
            var neg = Math.Exp(-x);

            return (pos - neg) / (pos + neg);
        }

        public static double Derivative(double x)
        {
            var o = Function(x);
            return DerivativeByOutput(o);
        }

        public static double DerivativeByOutput(double output)
        {
            return 1 - Math.Pow(output, 2);
        }
    }
}
