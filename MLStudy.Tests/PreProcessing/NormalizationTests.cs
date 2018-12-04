using MLStudy.PreProcessing;
using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace MLStudy.Tests.PreProcessing
{
    public class NormalizationTests
    {
        [Fact]
        public void MinMaxTest()
        {
            var data = new TensorOld(new double[] { 1, 2, 3, 4, 3, 2, 3, 4, 5, 7 });
            var norm = new MinMaxNorm(data);
            var test = new TensorOld(new double[] { 1, 7, 3, 4 });
            var result = norm.Normalize(test);

            Assert.Equal(1, norm.Min);
            Assert.Equal(7, norm.Max);
            Assert.Equal(0, result[0]);
            Assert.Equal(1, result[1]);
            Assert.Equal(2d/6d, result[2]);
            Assert.Equal(0.5, result[3]);
        }

        [Fact]
        public void ZScoreTest()
        {
            var data = new TensorOld(new double[] { 1, 2, 3 });
            var norm = new ZScoreNorm(data);
            var test = new TensorOld(new double[] { 1.5, 2.5 });
            var result = norm.Normalize(test);

            Assert.Equal(2, norm.Mean);
            Assert.Equal(Math.Sqrt(2d / 3d), norm.Delta);
            Assert.True(result[0] < 0);
            Assert.True(result[1] > 0);
        }
    }
}
