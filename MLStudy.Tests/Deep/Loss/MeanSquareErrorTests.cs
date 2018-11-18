using MLStudy.Deep;
using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace MLStudy.Tests.Deep
{
    public class MeanSquareErrorTests
    {
        [Fact]
        public void LossTest()
        {
            var y = new Tensor(new double[] { 1, 3, 2, 4, 5, 6 });
            var yHat = new Tensor(new double[] { 1.5, 2.6, 2.1, 3.9, 5.3, 6.7 });
            var error = MeanSquareError.Function(y, yHat);
            var expected = 0.084166666666666667;
            MyAssert.ApproximatelyEqual(expected, error);
        }

        [Fact]
        public void DerivativeTest()
        {
            var y = new Tensor(new double[] { 1, 3, 2, 4, 5, 6 });
            var yHat = new Tensor(new double[] { 1.5, 2.6, 2.1, 3.9, 5.3, 6.7 });
            var loss = MeanSquareError.Function(y, yHat);
            var gradient = MeanSquareError.Derivative(y, yHat);
            var delta = 0.00001;
            yHat[0] += delta;
            var expected = (MeanSquareError.Function(y, yHat) - loss) / delta;

            MyAssert.ApproximatelyEqual(expected, gradient[0]);
        }
    }
}
