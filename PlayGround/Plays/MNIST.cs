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
            var trainX = MNISTReader.ReadImagesToTensor4("Data\\train-images.idx3-ubyte", 60000);
            var trainY = MNISTReader.ReadLabelsToMatrix("Data\\train-labels.idx1-ubyte", 60000);
            var testX = MNISTReader.ReadImagesToTensor4("Data\\t10k-images.idx3-ubyte", 10000);
            var testY = MNISTReader.ReadLabelsToMatrix("Data\\t10k-labels.idx1-ubyte", 10000);

            var nn = new NeuralNetwork()
                .AddLayer(new ConvLayer(10, 5, 1))
                .AddLayer(new MaxPooling(2))
                .AddReLU()
                //.AddLayer(new ConvLayer(20, 5, 1))
                //.AddLayer(new MaxPooling(2))
                //.AddReLU()
                .AddLayer(new FlattenLayer())
                .AddFullLayer(100)
                .AddSigmoid()
                .AddFullLayer(50)
                .AddSigmoid()
                .AddFullLayer(10)
                .AddSoftmaxWithCrossEntropyLoss()
                .UseAdam();

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

            trainer.StartTrain(trainX, trainY, testX, testY);
        }
    }
}
