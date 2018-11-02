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
            var derivative = forwardOutput.ApplyFunction(DerivativeFunctions.SigmoidByResult);
            return MatrixOperations.Instance.MultipleElementWise(derivative, outputError);
        }

        public override Matrix Forward(Matrix input)
        {
            return input.ApplyFunction(Functions.Sigmoid);
        }
    }
}
