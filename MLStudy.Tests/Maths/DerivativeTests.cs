using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace MLStudy.Tests.Maths
{
    public class DerivativeTests
    {
        [Fact]
        public void ReLUTests()
        {
            var output = new Tensor(new double[] { 1, 2, 0, 5, 0, 7, 4, 6, 8, 0, 4, 1 }, 3, 4);
            var expected = new Tensor(new double[] { 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1 }, 3, 4);
            var actual = Tensor.Apply(output, Derivatives.ReLU);
            Assert.Equal(expected, actual);
        }

        [Fact]
        public void SigmoidTest()
        {
            var x = 0;
            var d = Derivatives.Sigmoid(0);
            var delta = 0.0001;
            var expected = (Functions.Sigmoid(x + delta) - Functions.Sigmoid(x)) / delta;
            MyAssert.ApproximatelyEqual(expected, d);
        }

        [Fact]
        public void SoftmaxTest()
        {
            var output = new double[] { 0.05, 0.15, 0.7, 0.1 };
            var expected = new double[,]
            { { 0.0475, -0.0075, -0.035, -0.005} ,
              { -0.0075, 0.1275, -0.105, -0.015},
              { -0.035, -0.105, 0.21, -0.07},
              { -0.005, -0.015, -0.07, 0.09 } };
            var actual = Derivatives.SoftmaxFromOutput(output);
            MyAssert.ApproximatelyEqual(expected, actual);
        }

        [Fact]
        public void TanhTest()
        {
            var x = 0;
            var d = Derivatives.Tanh(x);
            var delta = 0.0001;
            var expected = (Functions.Tanh(x + delta) - Functions.Tanh(x)) / delta;
            MyAssert.ApproximatelyEqual(expected, d);
        }
    }
}
