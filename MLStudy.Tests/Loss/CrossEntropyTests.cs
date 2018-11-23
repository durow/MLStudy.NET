using MLStudy.Deep;
using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace MLStudy.Tests.Deep
{
    public class CrossEntropyTests
    {
        [Fact]
        public void ForwardTest()
        {
            //模拟sigmoid的情况，4个样本
            var yHat = new Tensor(new double[] { 0.7, 0.2, 0.4, 0.9 }, 4, 1);
            var yHat2 = new Tensor(new double[] { 0.7, 0.2, 0.6, 0.9 }, 4, 1);
            var y = new Tensor(new double[] { 1, 0, 1, 1 }, 4, 1);

            var ce = new CrossEntropy();
            ce.PrepareTrain(y, yHat);

            ce.Compute(y, y);
            var minLoss = ce.GetLoss();

            ce.Compute(y, yHat);
            var loss = ce.GetLoss();

            ce.Compute(y, yHat2);
            var loss2 = ce.GetLoss();

            Assert.Equal(0, minLoss);
            Assert.True(loss > minLoss);
            Assert.True(loss > loss2);
        }

        [Fact]
        public void ForwardTest2()
        {
            var yHat = new Tensor(new double[] { 0.05, 0.15, 0.7, 0.1 }, 1, 4);
            var yHat2 = new Tensor(new double[] { 0.1, 0.15, 0.65, 0.1 }, 1, 4);
            var y = new Tensor(new double[] { 0, 0, 1, 0 }, 1, 4);

            var ce = new CrossEntropy();
            ce.PrepareTrain(y, yHat);

            ce.Compute(y, yHat);
            var loss = ce.GetLoss();

            ce.Compute(y, yHat2);
            var loss2 = ce.GetLoss();

            Assert.True(loss < loss2);
        }

        [Fact]
        public void AccuracyTest()
        {
            var y = new Tensor(new double[] { 1, 0, 1, 0, 1, 0 }, 6, 1);
            var yHat = new Tensor(new double[] { 0, 1, 1, 1, 1, 0 }, 6, 1);
            var expected = 0.5;
            var actual = CrossEntropy.ComputeAccuracy(y, yHat);
            Assert.Equal(expected, actual);
        }

        [Fact]
        public void AccuracyTest2()
        {
            var y = new Tensor(new double[] { 0, 0, 1, 0, 1, 0 }, 2, 3);
            var yHat = new Tensor(new double[] { 0, 1, 0, 0, 1, 0 }, 2, 3);
            var expected = 0.5;
            var actual = CrossEntropy.ComputeAccuracy(y, yHat);
            Assert.Equal(expected, actual);
        }
    }
}
