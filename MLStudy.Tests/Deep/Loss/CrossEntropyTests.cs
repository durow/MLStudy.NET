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
        public void FunctionTest()
        {
            var loss1 = CrossEntropy.Function(1, 0.7);
            var loss2 = CrossEntropy.Function(1, 0.9);
            Assert.True(loss1 > loss2);

            var loss3 = CrossEntropy.Function(new double[] { 0.2, 0.3, 0.5 }, new double[] { 0.2, 0.3, 0.5 });
            var loss4 = CrossEntropy.Function(new double[] { 0.2, 0.3, 0.5 }, new double[] { 0.3, 0.2, 0.5 });
            Assert.True(loss3 < loss4);
        }

        [Fact]
        public void GetLossTest()
        {
            //模拟sigmoid的情况，4个样本
            var ce = new CrossEntropy();
            var yHat = new Tensor(new double[] { 0.7, 0.2, 0.4, 0.9 }, 4, 1);
            var yHat2 = new Tensor(new double[] { 0.7, 0.2, 0.6, 0.9 }, 4, 1);
            var y = new Tensor(new double[] { 1, 0, 1, 1 }, 4, 1);
            var minLoss = ce.GetLoss(y, y);
            var loss = ce.GetLoss(y, yHat);
            var loss2 = ce.GetLoss(y, yHat2);

            Assert.Equal(0, minLoss);
            Assert.True(loss > minLoss);
            Assert.True(loss > loss2);
        }

        [Fact]
        public void GetLossTest2()
        {
            var ce = new CrossEntropy();
            var yHat = new Tensor(new double[] { 0.05, 0.15, 0.7, 0.1 }, 1, 4);
            var yHat2 = new Tensor(new double[] { 0.1, 0.15, 0.65, 0.1 }, 1, 4);
            var y = new Tensor(new double[] { 0, 0, 1, 0 }, 1, 4);

            var loss = ce.GetLoss(y, yHat);
            var loss2 = ce.GetLoss(y, yHat2);

            Assert.True(loss < loss2);
        }
    }
}
