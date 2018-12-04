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
            var input = new TensorOld(new double[] { 1, 2, -3, 5, -2, 7, 4, 6, 8, -5, 4, 1 }, 3, 4);
            var expected = new TensorOld(new double[] { 1, 2, 0, 5, 0, 7, 4, 6, 8, 0, 4, 1 }, 3, 4);
            var relu = new ReLU();
            relu.PrepareTrain(input);
            var actual = relu.Forward(input);
            Assert.Equal(expected, actual);

            var error = TensorOld.Values(0.1, 3, 4);
            var expError = new TensorOld(new double[] { 0.1, 0.1, 0, 0.1, 0, 0.1, 0.1, 0.1, 0.1, 0, 0.1, 0.1 }, 3, 4);
            var actError = relu.Backward(error);
            Assert.Equal(expError, actError);
        }
    }
}
