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

    public class Activation
    {
        private static Dictionary<Activations, Func<double, double>> activationDict = new Dictionary<Activations, Func<double, double>>();
        private static Dictionary<Activations, Func<double, double>> derivativeDict = new Dictionary<Activations, Func<double, double>>();
        public Activations ActivationType { get; set; }

        static Activation()
        {
            activationDict.Add(Activations.None, new Func<double, double>(a => a));
            activationDict.Add(Activations.ReLU, new Func<double, double>(Functions.ReLU));
            activationDict.Add(Activations.Sigmoid, new Func<double, double>(Functions.Sigmoid));

            derivativeDict.Add(Activations.None, new Func<double, double>(a => a));
            activationDict.Add(Activations.ReLU, new Func<double, double>(DerivativeFunctions.ReLU));
            activationDict.Add(Activations.Sigmoid, new Func<double, double>(DerivativeFunctions.SigmoidByResult));
        }

        public Activation(Activations activationType)
        {
            ActivationType = activationType;
        }

        public Matrix Derivative(Matrix outputError)
        {
            return outputError.ApplyFunction(derivativeDict[ActivationType]);
        }

        public Matrix Active(Matrix m)
        {
            return m.ApplyFunction(activationDict[ActivationType]);
        }

        public Vector Active(Vector v)
        {
            return v.ApplyFunction(activationDict[ActivationType]);
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
