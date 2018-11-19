using MLStudy.Deep;
using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace MLStudy.Tests.Deep
{
    public class ReLUTests
    {
        [Fact]
        public void ReLUTest()
        {
            var relu = new ReLU();
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
