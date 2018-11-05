using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Activations
{
    public class ReLU : Activation
    {
        public override string Name => "ReLU";

        public override Matrix Backward(Matrix forwardOutput, Matrix outputError)
        {
            var derivative = forwardOutput.ApplyFunction(DerivativeFunctions.ReLU);
            return TensorOperations.Instance.MultipleElementWise(derivative, outputError);
        }

        public override Tensor3 Backward(Tensor3 forwardOutput, Tensor3 outputError)
        {
            var derivative = forwardOutput.ApplyFunction(DerivativeFunctions.ReLU);
            return TensorOperations.Instance.MultipleElementWise(derivative, outputError);
        }

        public override Matrix Forward(Matrix input)
        {
            return input.ApplyFunction(Functions.ReLU);
        }

        public override Tensor3 Forward(Tensor3 input)
        {
            return input.ApplyFunction(Functions.ReLU);
        }
    }
}
