using MLStudy.Abstraction;

namespace MLStudy.Deep
{
    public class MeanSquareError: LossFunction
    {
        public double GetLoss(Tensor y, Tensor yHat)
        {
            return Function(y, yHat);
        }

        public Tensor GetGradient(Tensor y, Tensor yHat)
        {
            return Derivative(y, yHat);
        }

        public static double Function(Tensor y, Tensor yHat)
        {
            Tensor.CheckShape(y, yHat);

            var error = y - yHat;
            error = error.Reshape(error.ElementCount);
            var result = error * error / (2 * error.ElementCount);
            return result.GetValue();
        }

        public static Tensor Derivative(Tensor y, Tensor yHat)
        {
            Tensor.CheckShape(y, yHat);

            //因为存在learning rate，所以梯度前面的系数不那么重要，但最好和损失函数一致，
            return (yHat - y) / y.ElementCount;
        }

        public override void PrepareTrain(Tensor y, Tensor yHat)
        {
            throw new System.NotImplementedException();
        }

        public override void Compute(Tensor y, Tensor yHat)
        {
            throw new System.NotImplementedException();
        }
    }
}
