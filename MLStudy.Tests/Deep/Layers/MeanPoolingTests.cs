using MLStudy.Deep;
using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace MLStudy.Tests.Deep.Layers
{
    public class MeanPoolingTests
    {
        [Fact]
        public void ForwardBackwardTest()
        {
            var data = new double[]
            {
                3,3,1,5,
                4,2,6,4,
                8,2,6,3,
                6,4,9,6
            };
            var input = new TensorOld(data, 1, 1, 4, 4);
            var pooling = new MeanPooling(2);
            pooling.PrepareTrain(input);
            var forward = pooling.Forward(input);
            var expected = new TensorOld(new double[]
            {
                3,4,
                5,6
            }, 1, 1, 2, 2);

            Assert.Equal(expected, forward);

            var error = new TensorOld(new double[]
            {
                0.4, 0.8,
                -1.2, 1.6,
            }, 1, 1, 2, 2);
            var backward = pooling.Backward(error);
            var backExpected = new TensorOld(new double[]
            {
                0.1,0.1,0.2,0.2,
                0.1,0.1,0.2,0.2,
                -0.3,-0.3,0.4,0.4,
                -0.3,-0.3,0.4,0.4,
            }, 1, 1, 4, 4);
            Assert.Equal(backExpected, backward);
        }

        [Fact]
        public void ForwardBackwardTest2()
        {
            var data = new double[]
            {
                3,3,1,5,
                4,2,6,4,
                8,2,6,3,
                6,4,9,6,

                3,3,1,5,
                4,2,6,4,
                8,2,6,3,
                6,4,9,6,

                3,3,1,5,
                4,2,6,4,
                8,2,6,3,
                6,4,9,6,
            };
            var input = new TensorOld(data, 1, 3, 4, 4);
            var pooling = new MeanPooling(2);
            pooling.PrepareTrain(input);
            var actual = pooling.Forward(input);
            var expected = new TensorOld(new double[]
            {
                3,4,
                5,6,

                3,4,
                5,6,

                3,4,
                5,6,
            }, 1, 3, 2, 2);

            Assert.Equal(expected, actual);

            var error = new TensorOld(new double[]
            {
                0.4, 0.8,
                -1.2, 1.6,

                0.4, 0.8,
                -1.2, 1.6,

                0.4, 0.8,
                -1.2, 1.6,
            }, 1, 3, 2, 2);
            var backward = pooling.Backward(error);
            var backExpected = new TensorOld(new double[]
            {
                0.1,0.1,0.2,0.2,
                0.1,0.1,0.2,0.2,
                -0.3,-0.3,0.4,0.4,
                -0.3,-0.3,0.4,0.4,

                0.1,0.1,0.2,0.2,
                0.1,0.1,0.2,0.2,
                -0.3,-0.3,0.4,0.4,
                -0.3,-0.3,0.4,0.4,

                0.1,0.1,0.2,0.2,
                0.1,0.1,0.2,0.2,
                -0.3,-0.3,0.4,0.4,
                -0.3,-0.3,0.4,0.4,
            }, 1, 3, 4, 4);
            Assert.Equal(backExpected, backward);
        }

        [Fact]
        public void ForwardBackwardTest3()
        {
            var data = new double[]
            {
                3,2,3,5,
                4,2,6,1,
                8,2,6,3,
                6,5,9,5,

                3,2,3,5,
                4,2,6,1,
                8,2,6,3,
                6,5,9,5,

                3,2,3,5,
                4,2,6,1,
                8,2,6,3,
                6,5,9,5,

                3,2,3,5,
                4,2,6,1,
                8,2,6,3,
                6,5,9,5,

                3,2,3,5,
                4,2,6,1,
                8,2,6,3,
                6,5,9,5,

                3,2,3,5,
                4,2,6,1,
                8,2,6,3,
                6,5,9,5,
            };
            var input = new TensorOld(data, 2, 3, 4, 4);
            var pooling = new MeanPooling(1, 3, 1, 3);
            pooling.PrepareTrain(input);
            var actual = pooling.Forward(input);
            var expected = new TensorOld(new double[]
            {
                2,3,4,5,
                2,3,4,5,
                2,3,4,5,
                2,3,4,5,
                2,3,4,5,
                2,3,4,5,
            }, 2, 3, 4, 1);

            Assert.Equal(expected, actual);

            var error = new TensorOld(new double[]
            {
                0.8,0.4,-1.6,1.2,
                0.8,0.4,-1.6,1.2,
                0.8,0.4,-1.6,1.2,
                0.8,0.4,-1.6,1.2,
                0.8,0.4,-1.6,1.2,
                0.8,0.4,-1.6,1.2,
            }, 2, 3, 4, 1);
            var backward = pooling.Backward(error);

            var backData = new double[]
            {
                0.2,0.2,0.2,0,
                0.1,0.1,0.1,0,
                -0.4,-0.4,-0.4,0,
                0.3,0.3,0.3,0,

                0.2,0.2,0.2,0,
                0.1,0.1,0.1,0,
                -0.4,-0.4,-0.4,0,
                0.3,0.3,0.3,0,

                0.2,0.2,0.2,0,
                0.1,0.1,0.1,0,
                -0.4,-0.4,-0.4,0,
                0.3,0.3,0.3,0,

                0.2,0.2,0.2,0,
                0.1,0.1,0.1,0,
                -0.4,-0.4,-0.4,0,
                0.3,0.3,0.3,0,

                0.2,0.2,0.2,0,
                0.1,0.1,0.1,0,
                -0.4,-0.4,-0.4,0,
                0.3,0.3,0.3,0,

                0.2,0.2,0.2,0,
                0.1,0.1,0.1,0,
                -0.4,-0.4,-0.4,0,
                0.3,0.3,0.3,0,
            };
            var backExpected = new TensorOld(backData, 2, 3, 4, 4);
            Assert.Equal(backExpected, backward);
        }
    }
}
