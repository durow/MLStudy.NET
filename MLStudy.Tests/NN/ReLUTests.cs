using MLStudy.NN;
using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace MLStudy.Tests.NN
{
    public class ReLUTests
    {
        [Fact]
        public void FunctionTest()
        {
            var input = new Tensor(new double[] { 1, 2, -3, 5, -2, 7, 4, 6, 8, -5, 4, 1 }, 3, 4);
            var expected = new Tensor(new double[] { 1, 2, 0, 5, 0, 7, 4, 6, 8, 0, 4, 1 }, 3, 4);
            var actual = Tensor.Apply(input, ReLULayer.Function);
            Assert.Equal(expected, actual);
        }

        [Fact]
        public void DerivativeTest()
        {
            var output = new Tensor(new double[] { 1, 2, 0, 5, 0, 7, 4, 6, 8, 0, 4, 1 }, 3, 4);
            var expected = new Tensor(new double[] { 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1 }, 3, 4);
            var actual = Tensor.Apply(output, ReLULayer.Derivative);
            Assert.Equal(expected, actual);
        }

        [Fact]
        public void ReLUTest1()
        {
            var relu = new ReLULayer();
            var input = new Tensor(new double[] { 1, 2, -3, 5, -2, 7, 4, 6, 8, -5, 4, 1 }, 3, 4);
            var expected = new Tensor(new double[] { 1, 2, 0, 5, 0, 7, 4, 6, 8, 0, 4, 1 }, 3, 4);
            var actual = relu.Forward(input);
            Assert.Equal(expected, actual);

            var error = Tensor.Values(0.1, 3, 4);
            var expError = new Tensor(new double[] { 0.1, 0.1, 0, 0.1, 0, 0.1, 0.1, 0.1, 0.1, 0, 0.1, 0.1 }, 3, 4);
            var actError = relu.Backward(error);
            Assert.Equal(expError, actError);
        }
    }
}
