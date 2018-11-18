using MLStudy.Deep;
using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace MLStudy.Tests.Deep
{
    public class SigmoidTests
    {
        [Fact]
        public void FunctionTest()
        {
            var x = 0;
            var expected = 0.5;
            var actual = Sigmoid.Function(x);
            Assert.Equal(expected, actual);

            var x2 = 10; //10已经很大了，太容易饱和
            var o2 = Sigmoid.Function(x2);
            Assert.True(o2 < 1);

            var x3 = -10;
            var o3 = Sigmoid.Function(x2);
            Assert.True(o3 > 0);
        }

        [Fact]
        public void DerivativeTest()
        {
            var x = 0;
            var d = Sigmoid.Derivative(0);
            var delta = 0.0001;
            var expected = (Sigmoid.Function(x + delta) - Sigmoid.Function(x)) / delta;
            MyAssert.ApproximatelyEqual(expected, d);
        }
    }
}
