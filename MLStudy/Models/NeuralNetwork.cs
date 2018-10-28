using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class NeuralNetwork : ITrain
    {
        public double LearningRate { get; set; }
        public List<FullyConnectedLayer> HiddenLayers { get; private set; }
        public OutputLayer OutLayer { get; private set; }
        public int InputFeatures { get; private set; }
        public NeuralNetwork(int inputFeatures, OutputTypes outType, params int[] hiddenLayers)
        {
            InputFeatures = inputFeatures;
            OutLayer = new OutputLayer();
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

    public class Activation
    {
        private static Dictionary<Activations, Func<double, double>> dict = new Dictionary<Activations, Func<double, double>>();
        public Activations ActivationType { get; set; }

        static Activation()
        {
            dict.Add(Activations.None, new Func<double, double>(a => a));
            dict.Add(Activations.ReLU, new Func<double, double>(Functions.ReLU));
            dict.Add(Activations.Sigmoid, new Func<double, double>(Functions.Sigmoid));
        }

        public Activation(Activations activationType)
        {
            ActivationType = activationType;
        }

        public Matrix Active(Matrix m)
        {
            return m.ApplyFunction(dict[ActivationType]);
        }

        public Vector Active(Vector v)
        {
            return v.ApplyFunction(dict[ActivationType]);
        }
    }

    public enum Activations
    {
        ReLU,
        Tanh,
        Sigmoid,
        None,
    }

    public enum OutputTypes
    {
        Sigmoid,
        Regression,
        Softmax,
    }
}
