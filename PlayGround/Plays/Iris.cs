using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using MLStudy;
using MLStudy.Data;
using MLStudy.Deep;
using MLStudy.PreProcessing;

namespace PlayGround.Plays
{
    public class Iris : IPlay
    {
        public void Play()
        {
            var iris = AyxCsvReader.Instance.ReadCsvFileDataTable("Data/iris.csv");
            var labels = DataFormat.DataTableToList(iris, "Species").ToList();
            var codec = new OneHotCodec<string>(labels);
            var y = codec.Encode(labels);
            var X = DataFormat.DataTableToTensor(iris, 1, 2, 3, 4);

            var engine = new Network()
                .AddFullLayer(10)
                .AddSigmoid()
                .AddFullLayer(6)
                .AddSigmoid()
                .AddFullLayer(codec.Length)
                .AddSoftmax()
                .UseCrossEntropyLoss()
                .UseGradientDescent(0.01);

            var trainer = new Train(engine, 16, 100, true);
            trainer.StartTrain(X, y, null, null);

            var machine = new Machine<string>(engine, MachineType.Classification);
            machine.LabelCodec = codec;

            while (true)
            {
                Console.Write("input Sepal.Length:");
                var a1 = double.Parse(Console.ReadLine());
                Console.Write("input Sepal.Width:");
                var a2 = double.Parse(Console.ReadLine());
                Console.Write("input Petal.Length:");
                var a3 = double.Parse(Console.ReadLine());
                Console.Write("input Petal.Width:");
                var a4 = double.Parse(Console.ReadLine());

                var input = new Tensor(new double[] { a1, a2, a3, a4 }, 1, 4);
                Console.WriteLine($"you input is {machine.Predict(input).First()},  Codec:{machine.LastResultCodec},  Raw:{machine.LastRawResult}");
            }
        }
    }
}
