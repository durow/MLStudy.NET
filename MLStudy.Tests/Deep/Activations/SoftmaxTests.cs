using MLStudy.Deep;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Xunit;

namespace MLStudy.Tests.Deep
{
    public class SoftmaxTests
    {
        [Fact]
        public void ForwadBackwardTest()
        {
            var input = new Tensor(new double[] { 7, 9, 1, -1, 2, -7, 2, 4, 7, 8, 4, -1 }, 3, 4);
            var y = new Tensor(new double[] { 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0 }, 3, 4); //第1个样本正确，第2，3个样本错误
            var soft = new MLStudy.Deep.Softmax();
            soft.PrepareTrain(input);
            var output = soft.Forward(input);
            var error = Tensor.DivideElementWise(y, output) * -1;
            var actual = soft.Backward(error);
            var expected = output - y; //推导出来的结果，因为要把softmax和损失函数分离，所以实际应用时要分别计算

            //存在精度问题，有些值无法完全相等
            MyAssert.ApproximatelyEqual(expected, actual);
        }
    }
}
