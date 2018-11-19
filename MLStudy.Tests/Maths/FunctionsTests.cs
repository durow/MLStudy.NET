using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Xunit;

namespace MLStudy.Tests.Maths
{
    public class FunctionsTests
    {
        [Fact]
        public void ReLUTests()
        {
            var input = new Tensor(new double[] { 1, 2, -3, 5, -2, 7, 4, 6, 8, -5, 4, 1 }, 3, 4);
            var expected = new Tensor(new double[] { 1, 2, 0, 5, 0, 7, 4, 6, 8, 0, 4, 1 }, 3, 4);
            var actual = Tensor.Apply(input, Functions.ReLU);
            Assert.Equal(expected, actual);
        }

        [Fact]
        public void SigmoidTest()
        {
            var x = 0;
            var expected = 0.5;
            var actual = Functions.Sigmoid(x);
            Assert.Equal(expected, actual);

            var x2 = 10; //10已经很大了，太容易饱和
            var o2 = Functions.Sigmoid(x2);
            Assert.True(o2 < 1);

            var x3 = -10;
            var o3 = Functions.Sigmoid(x3);
            Assert.True(o3 > 0);
        }

        [Fact]
        public void SoftmaxTest()
        {
            var data = new double[] { 10, 8, -4, 9, -2, 5, 3, 6 };
            var tensor = new Tensor(data);
            var output = Functions.Softmax(data);

            Assert.True(output.Sum() > 0.9999);

            var max = output.Max();
            var min = output.Min();
            Assert.Equal(max, output[0]);
            Assert.Equal(min, output[2]);
        }
    }
}
