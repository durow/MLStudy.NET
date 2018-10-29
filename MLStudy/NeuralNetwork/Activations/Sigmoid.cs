using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Activations
{
    public class Sigmoid : Activation
    {
        public override Matrix Backward(Matrix forwardOutput, Matrix outputError)
        {
            var derivative = forwardOutput.ApplyFunction(DerivativeFunctions.SigmoidByResult);
            return Tensor.MultipleElementWise(derivative, outputError);
        }

        public override Matrix Forward(Matrix input)
        {
            return input.ApplyFunction(Functions.Sigmoid);
        }
    }
}
