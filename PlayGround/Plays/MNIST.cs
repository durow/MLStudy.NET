using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using MLStudy;
using MLStudy.Abstraction;
using MLStudy.Data;
using MLStudy.Deep;
using MLStudy.Optimization;
using MLStudy.PreProcessing;

namespace PlayGround.Plays
{
    public class MNIST : IPlay
    {
        public void Play()
        {
            var trainX = MNISTReader.ReadImagesToMatrix("Data\\train-images.idx3-ubyte", 60000);
            var trainY = MNISTReader.ReadLabelsToMatrix("Data\\train-labels.idx1-ubyte", 60000);

            var nn = new NeuralNetwork()
                .AddFullLayer(50)
                .AddSigmoid()
                .AddFullLayer(10)
                .AddSoftmax()
                .UseOptimizer(new Adam())
                .UseCrossEntropyLoss();

            var cate = trainY
                .GetRawValues()
                .Distinct()
                .OrderBy(a => a)
                .Select(a => a.ToString())
                .ToList();

            var codec = new OneHotCodec(cate);
            var norm = new ZScoreNorm(trainX);

            var trainer = new Trainer(nn, 64, 10, true)
            {
                LabelCodec = codec,
                Normalizer = norm,
            };

            trainer.StartTrain(trainX, trainY, null, null);
        }
    }
}
