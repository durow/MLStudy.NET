using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Activations
{
    public class Tanh : Activation
    {
        public override string Name => "Tanh";

        public override Matrix Backward(Matrix forwardOutput, Matrix outputError)
        {
            var derivative = forwardOutput.ApplyFunction(Derivatives.TanhByResult);
            return TensorOperations.Instance.MultipleElementWise(derivative, outputError);
        }

        public override Tensor3 Backward(Tensor3 forwardOutput, Tensor3 outputError)
        {
            var derivative = forwardOutput.ApplyFunction(Derivatives.TanhByResult);
            return TensorOperations.Instance.MultipleElementWise(derivative, outputError);
        }

        public override Matrix Forward(Matrix input)
        {
            return input.ApplyFunction(Functions.Tanh);
        }

        public override Tensor3 Forward(Tensor3 input)
        {
            return input.ApplyFunction(Functions.Tanh);
        }
    }
}
