using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class NeuralNetwork : ITrainable
    {
        public double LearningRate { get; set; }
        public List<FullyConnectedLayer> HiddenLayers { get; private set; }
        public OutputLayer OutLayer { get; private set; }
        public int InputFeatures { get; private set; }
        public NeuralNetwork(int inputFeatures, OutputTypes outType, params int[] hiddenLayers)
        {
            InputFeatures = inputFeatures;
            InitHiddenLayers(hiddenLayers);
        }

        private void InitHiddenLayers(int[] hiddenLayers)
        {
            HiddenLayers = new List<FullyConnectedLayer>();
            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                if (i == 0)
                    HiddenLayers.Add(new FullyConnectedLayer(InputFeatures, hiddenLayers[i]));
                else
                    HiddenLayers.Add(new FullyConnectedLayer(hiddenLayers[i - 1], hiddenLayers[i]));
            }
        }

        public void Step(Matrix X, Vector y)
        {
            Forward(X);
            Backward(y);
        }

        private void Forward(Matrix X)
        {
            Matrix input = X;

            for (int i = 0; i < HiddenLayers.Count; i++)
            {
                input = HiddenLayers[i].Forward(input);
            }

            OutLayer.Forward(input);
        }

        private void Backward(Vector y)
        {
            var outputError = OutLayer.Backward(y);

            for (int i = 0; i < HiddenLayers.Count; i++)
            {
                var l = HiddenLayers[HiddenLayers.Count - i - 1];
                outputError = l.Backward(outputError);
            }
        }

        public double Error(Vector yHat, Vector y)
        {
            throw new NotImplementedException();
        }

        public double Loss(Matrix X, Vector y)
        {
            throw new NotImplementedException();
        }
    }

    

    

    public enum OutputTypes
    {
        Sigmoid,
        Regression,
        Softmax,
    }
}
