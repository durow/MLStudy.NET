using MLStudy.Abstraction;
using System;
using System.Linq;
using System.Threading.Tasks;

namespace MLStudy.NN
{
    public class SoftmaxLayer : ILayer
    {
        public Tensor LastForwardOutput { get; private set; }

        public Tensor Backward(Tensor error)
        {
            if (error.Rank == 1)
                error.Reshape(1, error.Shape[0]);

            if (error.Rank != 2)
                throw new TensorShapeException("Softmax BP must have error.Rank=2");

            Tensor.CheckShape(LastForwardOutput, error);

            var result = LastForwardOutput.GetSameShape();

            Parallel.For(0, result.Shape[0], i =>
            {
                //这个方法直接将计算结果写入result，不需要开辟中间内存
                ErrorBP(LastForwardOutput, error, result, i);

                //以下用显示的链式法则，过程更明显
                //var jacob = Derivative(LastForwardOutput.GetTensorByDim1(i)); //求得单个样本输出的softmax的导数
                //var gradient = error.GetTensorByDim1(i) * jacob;  //导数乘上反向传播过来的误差
                //for (int j = 0; j < result.Shape[1]; j++)  //结果写入result
                //{
                //    result[i, j] = gradient[0,j];
                //}
            });

            return result;
        }

        public Tensor Forward(Tensor input)
        {
            if(input.Rank == 1)
                input.Reshape(1, input.Shape[0]);

            if (input.Rank != 2)
                throw new TensorShapeException("Softmax input must have input.Rank=2");

            LastForwardOutput = input.GetSameShape();
            var sampleNum = input.Shape[0];
            for (int i = 0; i < sampleNum; i++)
            {
                var output = Function(input.GetTensorByDim1(i));
                for (int j = 0; j < output.ElementCount; j++)
                {
                    LastForwardOutput[i, j] = output[j];
                }
            }
            return LastForwardOutput;
        }

        public static Tensor Function(Tensor x)
        {
            if (x.Rank != 1)
                throw new TensorShapeException("Softmax input must x.Rank=1");

            var max = x.Max();
            x -= max;
            x.Apply(a => Math.Exp(a));
            var sum = x.Sum();
            return x.Divide(sum);
        }

        private static void ErrorBP(Tensor output, Tensor error, Tensor result, int sampleIndex)
        {
            var column = output.Shape[1];
            for (int i = 0; i < column; i++)
            {
                var derivative = 0d;
                for (int j = 0; j < column; j++)
                {
                    if (i == j)
                        derivative += output[sampleIndex, i] * (1 - output[sampleIndex, j]) * error[sampleIndex, j];
                    else
                        derivative += -output[sampleIndex, i] * output[sampleIndex, j] * error[sampleIndex, j];
                }
                result[sampleIndex, i] = derivative;
            }
        }

        public static double[] Function(double[] x)
        {
            var result = new double[x.Length];

            var max = x.Max();
            for (int i = 0; i < x.Length; i++)
            {
                result[i] = Math.Exp(x[i] - max);
            }

            var sum = result.Sum();
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = result[i] / sum;
            }

            return result;
        }

        public static double[,] Derivative(double[] output)
        {
            var len = output.Length;
            var jacob = new double[len, len];
            for (int i = 0; i < len; i++)
            {
                for (int j = 0; j < len; j++)
                {
                    if (i == j)
                        jacob[i, j] = output[i] * (1 - output[j]);
                    else
                        jacob[i, j] = -output[i] * output[j];
                }
            }
            return jacob;
        }

        public static Tensor Derivative(Tensor output)
        {
            if (output.Rank != 1)
                throw new TensorShapeException("Softmax derivative output.Rank must be 1");

            var len = (int)output.ElementCount;
            var jacob = new Tensor(len,len);
            for (int i = 0; i < len; i++)
            {
                for (int j = 0; j < len; j++)
                {
                    if (i == j)
                        jacob[i, j] = output[i] * (1 - output[j]);
                    else
                        jacob[i, j] = -output[i] * output[j];
                }
            }
            return jacob;
        }
    }
}
