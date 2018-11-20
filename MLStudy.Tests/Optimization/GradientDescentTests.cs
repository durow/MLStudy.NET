using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace MLStudy.Tests.Optimization
{
    public class GradientDescentTests
    {
        [Fact]
        public void OptimizeTest()
        {
            var target = new Tensor(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, 3, 3);
            var gradient = new Tensor(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, 3, 3);
            var expected = new Tensor(new double[] { 0.9, 1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2, 8.1 }, 3, 3);
            var gd = new GradientDescent(0.1);
            gd.Optimize(target, gradient);
            Assert.Equal(expected, target);
        }
    }
}
