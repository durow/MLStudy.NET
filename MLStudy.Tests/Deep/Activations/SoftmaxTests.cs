using Xunit;

namespace MLStudy.Tests.Deep
{
    public class SoftmaxTests
    {
        [Fact]
        public void ForwadBackwardTest()
        {
            var soft = new MLStudy.Deep.Softmax();
            soft.PrepareTrain(new Tensor(3, 4));

            //模拟之前的N次操作
            for (int i = 0; i < 3; i++)
            {
                var noiseIn = Tensor.Rand(3, 4);
                var noiseError = Tensor.Rand(3, 4);
                soft.Forward(noiseIn);
                soft.Backward(noiseError);
            }

            //真正的测试开始，等正常输出说明不受前面影响
            var input = new Tensor(new double[] { 7, 9, 1, -1, 2, -7, 2, 4, 7, 8, 4, -1 }, 3, 4);
            var y = new Tensor(new double[] { 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0 }, 3, 4); //第1个样本正确，第2，3个样本错误
            var output = soft.Forward(input);
            //用交叉熵计算Loss时反向传回的error
            var error = Tensor.DivideElementWise(y, output) * -1;
            //计算出来的结果
            var actual = soft.Backward(error);
            //推导出来的结果，因为要把softmax和损失函数分离，所以实际应用时要分别计算
            var expected = output - y; 

            //存在精度问题，有些值无法完全相等
            MyAssert.ApproximatelyEqual(expected, actual);
        }
    }
}
