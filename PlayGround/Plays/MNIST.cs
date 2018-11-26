using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using MLStudy;
using MLStudy.Abstraction;
using MLStudy.Data;
using MLStudy.Deep;
using MLStudy.PreProcessing;

namespace PlayGround.Plays
{
    public class MNIST : IPlay
    {
        public void Play()
        {
            var trainX = MNISTReader.ReadImagesToMatrix("Data\\train-images.idx3-ubyte", 6000);
            var trainY = MNISTReader.ReadLabelsToMatrix("Data\\train-labels.idx1-ubyte", 6000);

            var nn = new NeuralNetwork()
                .AddFullLayer(20)
                .AddReLU()
                .AddFullLayer(10)
                .AddSoftmax()
                .UseGradientDescent(0.5)
                .UseCrossEntropyLoss();

            var cate = trainY.GetRawValues().Select(a => a.ToString()).ToList();
            var codec = new OneHotCodec<string>(cate);
            var y = codec.Encode(cate);

            //var norm = new ZScoreNorm(trainX - 128);
            var X = (trainX-128) / (255);

            var trainer = new Trainer(nn, 50, 10);
            trainer.StartTrain(X, y, null, null);

            
        }
    }
}
