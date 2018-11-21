using MLStudy;
using MLStudy.Deep;
using System;
using System.Collections.Generic;
using System.Text;

namespace PlayGround
{
    public class XOR : IPlay
    {
        public void Play()
        {
            Console.WriteLine("XOR Learn!");

            var engine = new Network()
                .AddFullLayer(2)
                .AddSigmoid()
                .AddFullLayer(2)
                .AddSigmoid()
                .AddFullLayer(1)
                .AddSigmoid()
                .UseCrossEntropyLoss()
                .UseGradientDescent(0.01)
                .UseRidge(0.0001);

            var (trainX, trainY) = GetXorData(1000);
            var (testX, testY) = GetXorData(500);

            var tain = new Train(engine, 32, 200);
            tain.StartTrain(trainX, trainY, testX, testY);

            Console.WriteLine("Complete!");
            var machine = new Machine(engine);

            while (true)
            {
                var str1 = Console.ReadLine();
                var str2 = Console.ReadLine();
                if (str1 == "" || str2 == "")
                    return;

                var a1 = double.Parse(str1);
                var a2 = double.Parse(str2);

                var result = machine.Predict(new Tensor(new double[] { a1, a2 }, 1, 2));

                Console.WriteLine(string.Join(",", result));
            }
        }

        public void Print(Tensor x, Tensor y)
        {
            for (int i = 0; i < x.Shape[0]; i++)
            {
                Console.WriteLine($"{x.GetTensorByDim1(i).ToString()}\t\t{y.GetTensorByDim1(i).ToString()}");
            }
        }

        public (Tensor,Tensor) GetXorData(int count)
        {
            var xbuff = DataEmulator.Instance.RandomArray(count, 2);
            var x = new Tensor(xbuff);
            var y = new Tensor(count, 1);
            x.Apply(a => a * 12 - 6);
            for (int i = 0; i < count; i++)
            {
                if (x[i, 0] * x[i, 1] > 0)
                    y[i,0] = 0;
                else
                    y[i,0] = 1;
            }
            return (x, y);
        }
    }
}
