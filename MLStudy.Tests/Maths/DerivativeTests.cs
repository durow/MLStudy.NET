using Xunit;

namespace MLStudy.Tests.Maths
{
    public class DerivativeTests
    {
        [Fact]
        public void ReLUTests()
        {
            var output = new Tensor(new double[] { 1, 2, 0, 5, 0, 7, 4, 6, 8, 0, 4, 1 }, 3, 4);
            var expected = new Tensor(new double[] { 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1 }, 3, 4);
            var actual = Tensor.Apply(output, Derivatives.ReLU);
            Assert.Equal(expected, actual);
        }

        [Fact]
        public void SigmoidTest()
        {
            var x = 0;
            var d = Derivatives.Sigmoid(0);
            var delta = 0.0001;
            var expected = (Functions.Sigmoid(x + delta) - Functions.Sigmoid(x)) / delta;
            MyAssert.ApproximatelyEqual(expected, d);
        }

        [Fact]
        public void SoftmaxTest()
        {
            var output = new double[] { 0.05, 0.15, 0.7, 0.1 };
            var expected = new double[,]
            { { 0.0475, -0.0075, -0.035, -0.005} ,
              { -0.0075, 0.1275, -0.105, -0.015},
              { -0.035, -0.105, 0.21, -0.07},
              { -0.005, -0.015, -0.07, 0.09 } };
            var actual = Derivatives.SoftmaxFromOutput(output);
            MyAssert.ApproximatelyEqual(expected, actual);
        }

        [Fact]
        public void TanhTest()
        {
            var x = 0;
            var d = Derivatives.Tanh(x);
            var delta = 0.0001;
            var expected = (Functions.Tanh(x + delta) - Functions.Tanh(x)) / delta;
            MyAssert.ApproximatelyEqual(expected, d);
        }

        [Fact]
        public void SigmoidCrossEntropyTest()
        {
            var y = new double[] { 1 };
            var yHat = new double[] { 0.7 };
            var der = Derivatives.CrossEntropy(y, yHat);

            var delta = 0.00001;
            var ce0 = Functions.CrossEntropy(y, yHat);
            yHat[0] += delta;
            var ce1 = Functions.CrossEntropy(y, yHat);
            var expected = (ce1 - ce0) / delta;

            MyAssert.ApproximatelyEqual(expected, der[0], 0.0001);
        }

        [Fact]
        public void SigmoidCrossEntropyTest2()
        {
            var y = new double[] { 0 };
            var yHat = new double[] { 0.3 };
            var der = Derivatives.CrossEntropy(y, yHat);

            var delta = 0.00001;
            var ce0 = Functions.CrossEntropy(y, yHat);
            yHat[0] += delta;
            var ce1 = Functions.CrossEntropy(y, yHat);
            var expected = (ce1 - ce0) / delta;

            MyAssert.ApproximatelyEqual(expected, der[0], 0.0001);
        }

        [Fact]
        public void SoftmaxCrossEntropyTest()
        {
            var y = new double[] { 0, 1, 0, 0 };
            var yHat = new double[] { 0.1, 0.7, 0.15, 0.05 };
            var der = Derivatives.CrossEntropy(y, yHat);

            var delta = 0.00001;
            var ce0 = Functions.CrossEntropy(y, yHat);
            yHat[0] += delta;
            var ce1 = Functions.CrossEntropy(y, yHat);
            var expected = (ce1 - ce0) / delta;

            MyAssert.ApproximatelyEqual(expected, der[0], 0.0001);
        }

        [Fact]
        public void SoftmaxCrossEntropyTest2()
        {
            var y = new double[] { 0, 1, 0, 0 };
            var yHat = new double[] { 0.1, 0.7, 0.15, 0.05 };
            var der = Derivatives.CrossEntropy(y, yHat);

            var delta = 0.00001;
            var ce0 = Functions.CrossEntropy(y, yHat);
            yHat[1] += delta;
            var ce1 = Functions.CrossEntropy(y, yHat);
            var expected = (ce1 - ce0) / delta;

            MyAssert.ApproximatelyEqual(expected, der[1], 0.0001);
        }

        [Fact]
        public void CrossEntropyTest()
        {
            var y = new double[] { 0.1, 0.7, 0.15, 0.05 };
            var yHat = new double[] { 0.2, 0.6, 0.15, 0.05 };
            var der = Derivatives.CrossEntropy(y, yHat);

            var delta = 0.00001;
            var ce0 = Functions.CrossEntropy(y, yHat);
            yHat[3] += delta;
            var ce1 = Functions.CrossEntropy(y, yHat);
            var expected = (ce1 - ce0) / delta;

            MyAssert.ApproximatelyEqual(expected, der[3], 0.0001);
        }

        [Fact]
        public void MeanSquareErrorTest()
        {
            var y = new Tensor(new double[] { 1, 3, 2, 4, 5, 6 });
            var yHat = new Tensor(new double[] { 1.5, 2.6, 2.1, 3.9, 5.3, 6.7 });
            var loss = Functions.MeanSquareError(y, yHat);
            var gradient = Derivatives.MeanSquareError(y, yHat);
            var delta = 0.00001;
            yHat[0] += delta;
            var expected = (Functions.MeanSquareError(y, yHat) - loss) / delta;

            MyAssert.ApproximatelyEqual(expected, gradient[0]);
        }
    }
}
