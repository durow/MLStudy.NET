using MLStudy.Abstraction;
using System;

namespace MLStudy.Deep
{
    public class CrossEntropy : LossFunction
    {
        private double[] yBuff;
        private double[] yHatBuff;
        private double[] derBuff;
        private int sampleNumber;

        public override void PrepareTrain(Tensor y, Tensor yHat)
        {
            Tensor.CheckShape(y, yHat);
            if (y.Rank != 2)
                throw new TensorShapeException("y and yHat must Rank=2");

            ForwardOutput = new Tensor(y.Shape[0]);
            BackwardOutput = yHat.GetSameShape();
            yBuff = new double[y.Shape[1]];
            yHatBuff = new double[y.Shape[1]];
            derBuff = new double[y.Shape[1]];
            sampleNumber = y.Shape[0];
        }

        public override void Compute(Tensor y, Tensor yHat)
        {
            var outData = ForwardOutput.GetRawValues();
            var derData = BackwardOutput.GetRawValues();

            for (int i = 0; i < sampleNumber; i++)
            {
                y.GetByDim1(i, yBuff);
                yHat.GetByDim1(i, yHatBuff);
                outData[i] = Functions.CrossEntropy(yBuff, yHatBuff);
                Derivatives.CrossEntropy(yBuff, yHatBuff, derBuff);
                Array.Copy(derBuff, 0, derData, i * derBuff.Length, derBuff.Length);
            }
        }
    }
}
