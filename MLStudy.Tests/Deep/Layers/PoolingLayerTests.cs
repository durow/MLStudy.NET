using MLStudy.Deep;
using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace MLStudy.Tests.Deep.Layers
{
    public class PoolingLayerTests
    {
        [Fact]
        public void ForwardBackwardTest()
        {
            var data = new double[]
            {
                3,2,1,5,
                4,2,6,1,
                8,2,6,3,
                6,3,9,5
            };
            var input = new Tensor(data, 1, 1, 4, 4);
            var pooling = new MaxPooling(2);
            pooling.PrepareTrain(input);
            var forward = pooling.Forward(input);
            var expected = new Tensor(new double[]
            {
                4,6,
                8,9
            }, 1, 1, 2, 2);

            Assert.Equal(expected, forward);

            var error = new Tensor(new double[] 
            {
                0.5, 0.7,
                -0.3, 1.2
            }, 1, 1, 2, 2);
            var backward = pooling.Backward(error);
            var backExpected = new Tensor(new double[]
            {
                0,0,0,0,
                0.5,0,0.7,0,
                -0.3,0,0,0,
                0,0,1.2,0
            }, 1, 1, 4, 4);
            Assert.Equal(backExpected, backward);
        }

        [Fact]
        public void ForwardTest2()
        {
            var data = new double[]
            {
                3,2,1,5,
                4,2,6,1,
                8,2,6,3,
                6,3,9,5,

                3,2,1,5,
                4,2,6,1,
                8,2,6,3,
                6,3,9,5,

                3,2,1,5,
                4,2,6,1,
                8,2,6,3,
                6,3,9,5,
            };
            var input = new Tensor(data, 1, 3, 4, 4);
            var pooling = new MaxPooling(2);
            pooling.PrepareTrain(input);
            var actual = pooling.Forward(input);
            var expected = new Tensor(new double[]
            {
                4,6,
                8,9,

                4,6,
                8,9,

                4,6,
                8,9,
            }, 1, 3, 2, 2);

            Assert.Equal(expected, actual);

            var error = new Tensor(new double[]
            {
                0.5, 0.7,
                -0.3, 1.2,

                0.5, 0.7,
                -0.3, 1.2,

                0.5, 0.7,
                -0.3, 1.2,
            }, 1, 3, 2, 2);
            var backward = pooling.Backward(error);
            var backExpected = new Tensor(new double[]
            {
                0,0,0,0,
                0.5,0,0.7,0,
                -0.3,0,0,0,
                0,0,1.2,0,

                0,0,0,0,
                0.5,0,0.7,0,
                -0.3,0,0,0,
                0,0,1.2,0,

                0,0,0,0,
                0.5,0,0.7,0,
                -0.3,0,0,0,
                0,0,1.2,0,
            }, 1, 3, 4, 4);
            Assert.Equal(backExpected, backward);
        }

        [Fact]
        public void ForwardTest3()
        {
            var data = new double[]
            {
                3,2,1,5,
                4,2,6,1,
                8,2,6,3,
                6,3,9,5,

                3,2,1,5,
                4,2,6,1,
                8,2,6,3,
                6,3,9,5,

                3,2,1,5,
                4,2,6,1,
                8,2,6,3,
                6,3,9,5,

                3,2,1,5,
                4,2,6,1,
                8,2,6,3,
                6,3,9,5,

                3,2,1,5,
                4,2,6,1,
                8,2,6,3,
                6,3,9,5,

                3,2,1,5,
                4,2,6,1,
                8,2,6,3,
                6,3,9,5,
            };
            var input = new Tensor(data, 2, 3, 4, 4);
            var pooling = new MaxPooling(1, 3, 1, 3);
            pooling.PrepareTrain(input);
            var actual = pooling.Forward(input);
            var expected = new Tensor(new double[]
            {
                3,6,8,9,
                3,6,8,9,
                3,6,8,9,
                3,6,8,9,
                3,6,8,9,
                3,6,8,9,
            }, 2, 3, 4, 1);

            Assert.Equal(expected, actual);

            var error = new Tensor(new double[]
            {
                0.8,0.7,-1.5,0.4,
                0.8,0.7,-1.5,0.4,
                0.8,0.7,-1.5,0.4,
                0.8,0.7,-1.5,0.4,
                0.8,0.7,-1.5,0.4,
                0.8,0.7,-1.5,0.4,
            }, 2, 3, 4, 1);
            var backward = pooling.Backward(error);

            var backData = new double[]
            {
                0.8,0,0,0,
                0,0,0.7,0,
                -1.5,0,0,0,
                0,0,0.4,0,

                0.8,0,0,0,
                0,0,0.7,0,
                -1.5,0,0,0,
                0,0,0.4,0,

                0.8,0,0,0,
                0,0,0.7,0,
                -1.5,0,0,0,
                0,0,0.4,0,

                0.8,0,0,0,
                0,0,0.7,0,
                -1.5,0,0,0,
                0,0,0.4,0,

                0.8,0,0,0,
                0,0,0.7,0,
                -1.5,0,0,0,
                0,0,0.4,0,

                0.8,0,0,0,
                0,0,0.7,0,
                -1.5,0,0,0,
                0,0,0.4,0,
            };
            var backExpected = new Tensor(backData, 2, 3, 4, 4);
            Assert.Equal(backExpected, backward);
        }
    }
}
