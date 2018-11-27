using MLStudy.Deep;
using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace MLStudy.Tests.Deep
{
    public class FlattenLayerTests
    {
        [Fact]
        public void FlattenTest()
        {
            var fl = new FlattenLayer();
            var input = new Tensor(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 }, 2, 1, 2, 2);
            var expected = new Tensor(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, }, 2, 4);
            var actual = fl.Forward(input);
            Assert.Equal(expected, actual);

            var back = fl.Backward(actual);
            Assert.Equal(input, back);
        }
    }
}
