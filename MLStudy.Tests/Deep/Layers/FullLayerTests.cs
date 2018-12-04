using MLStudy.Deep;
using Xunit;

namespace MLStudy.Tests.Deep
{
    public class FullLayerTests
    {
        [Fact]
        public void ForwardBackwardTest()
        {
            var fl = new FullLayer(3);
            var weights = TensorOld.Ones(4, 3);
            var bias = TensorOld.Values(0.5, 1, 3);
            var input = new TensorOld(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 }, 2, 4);
            var error = new TensorOld(new double[] { -1, 0.8, 1.5, 1, -1, 1 }, 2, 3);

            var forward = new TensorOld(new double[] { 10.5, 10.5, 10.5, 26.5, 26.5, 26.5 }, 2, 3);
            var backward = new TensorOld(new double[] { 1.3, 1.3, 1.3, 1.3, 1, 1, 1, 1 }, 2, 4);
            var biasGradient = new TensorOld(new double[] { 0, -0.2, 2.5}, 1, 3);
            var weightsGradient = new TensorOld(new double[,]
            { { 4,-4.2,6.5},
              { 4,-4.4,9},
              { 4,-4.6,11.5},
              { 4,-4.8,14}, });

            fl.PrepareTrain(input);
            fl.SetWeights(weights);
            fl.SetBias(bias);
            fl.Forward(input);
            fl.Backward(error);

            Assert.Equal(forward, fl.ForwardOutput);
            Assert.Equal(backward, fl.BackwardOutput);
            MyAssert.ApproximatelyEqual(biasGradient, fl.BiasGradient);
            Assert.Equal(weightsGradient, fl.WeightsGradient);

        }
    }
}
