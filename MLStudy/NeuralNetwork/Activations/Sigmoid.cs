using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Activations
{
    public class Sigmoid : Activation
    {
        public override string Name => "Sigmoid";

        public override Matrix Backward(Matrix forwardOutput, Matrix outputError)
        {
            var derivative = forwardOutput.ApplyFunction(Derivatives.SigmoidByResult);
            return TensorOperations.Instance.MultipleElementWise(derivative, outputError);
        }

        public override Tensor3 Backward(Tensor3 forwardOutput, Tensor3 outputError)
        {
            var derivative = forwardOutput.ApplyFunction(Derivatives.SigmoidByResult);
            return TensorOperations.Instance.MultipleElementWise(derivative, outputError);
        }

        public override Matrix Forward(Matrix input)
        {
            return input.ApplyFunction(Functions.Sigmoid);
        }

        public override Tensor3 Forward(Tensor3 input)
        {
            return input.ApplyFunction(Functions.Sigmoid);
        }
    }
}
