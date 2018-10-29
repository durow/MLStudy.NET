using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class LogisticRegressionOut : LinearRegressionOut
    {
        public LogisticRegressionOut(int inputFeatures):base(inputFeatures)
        { }

        protected override void ComputeOutputError(Vector y)
        {
            var yHat = ForwardOutput.ToVector();
            Loss = LossFunctions.LogisticError(yHat, y);
        }

        public override Matrix Forward(Matrix input)
        {
            base.Forward(input);
            ForwardOutput = ForwardOutput.ApplyFunction(Functions.Sigmoid);
            return ForwardOutput;
        }
    }
}
