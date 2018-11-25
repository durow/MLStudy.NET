using MLStudy.Deep;
using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace MLStudy.Tests.Deep
{
    public class ConvLayerTests
    {
        [Fact]
        public void ForwardBackwardTest()
        {
            var input = new Tensor(new double[]
            {
                4,6,1,4,
                8,4,5,1,
                5,3,5,7,
                1,7,2,8,
            }, 1, 1, 4, 4);

            var conv = new ConvLayer(4, 2, 1, 1);
            conv.SetFilters(new Tensor(new double[]
            {
                1,1,0,0,
                0,0,1,1,
                1,0,1,0,
                0,1,0,1,
            }, 4, 1, 2, 2));

            var expected = new Tensor(new double[]
            {
                0,0,0,0,0,
                4,10,7,5,4,
                8,12,9,6,1,
                5,8,8,12,7,
                1,8,9,10,8,

                4,10,7,5,4,
                8,12,9,6,1,
                5,8,8,12,7,
                1,8,9,10,8,
                0,0,0,0,0,

                0,4,6,1,4,
                0,12,10,6,5,
                0,13,7,10,8,
                0,6,10,7,15,
                0,1,7,2,8,

                4,6,1,4,0,
                12,10,6,5,0,
                13,7,10,8,0,
                6,10,7,15,0,
                1,7,2,8,0,
            }, 1, 4, 5, 5);

            conv.PrepareTrain(input);
            var acutal = conv.Forward(input);
            Assert.Equal(acutal, expected);

            var error = expected / 10;
            var back = conv.Backward(error);
        }
    }
}
