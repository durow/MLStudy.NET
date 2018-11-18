using MLStudy.Deep;
using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace MLStudy.Tests.Deep
{
    public class TanhTests
    {
        [Fact]
        public void FunctionTest()
        {
            var x = 0;
            var o = Tanh.Function(x);
            Assert.Equal(0, o);

            var x2 = 10;
            var o2 = Tanh.Function(x2);
            Assert.True(o2 < 1);

            var x3 = -10;
            var o3 = Tanh.Function(x3);
            Assert.True(o3 > -1);
        }

        [Fact]
        public void DerivativeTest()
        {
            var x = 0;
            var d = Tanh.Derivative(x);
            var delta = 0.0001;
            var expected = (Tanh.Function(x+delta) - Tanh.Function(x)) / delta;
            MyAssert.ApproximatelyEqual(expected, d);
        }
    }
}
