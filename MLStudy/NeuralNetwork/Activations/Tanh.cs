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
            var derivative = forwardOutput.ApplyFunction(DerivativeFunctions.TanhByResult);
            return MatrixOperations.Instance.MultipleElementWise(derivative, outputError);
        }

        public override Matrix Forward(Matrix input)
        {
            return input.ApplyFunction(Functions.Tanh);
        }
    }
}
