using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class FullyConnectedLayer
    {
        private DataEmulator emu = new DataEmulator();
        public Activation Activation { get; private set; }
        public Matrix Weights { get; private set; }
        public Vector Bias { get; private set; }
        public int NeuronCount { get; private set; }
        public int InputFeatures { get; private set; }
        public double LearningRate { get; set; }

        public Matrix ForwardInput { get; private set; }
        public Matrix ForwardLinear { get; private set; }
        public Matrix ForwardOutput { get; private set; }
        //d Loss/d Input
        public Matrix InputError { get; private set; }
        //d Loss/d z(Linear Output)
        public Matrix LinearError { get; private set; }
        //d Loss/d a(Activator Output)
        public Matrix OutputError { get; set; }

        public FullyConnectedLayer(int inputFeatures, int neuronCount)
        {
            InputFeatures = inputFeatures;
            NeuronCount = neuronCount;
            AutoInitWeightsBias();
        }

        public FullyConnectedLayer UseActivation(ActivationTypes activation)
        {
            UseActivation(activation.ToString());
            return this;
        }

        public FullyConnectedLayer UseActivation(string activationName)
        {
            Activation = Activation.Get(activationName);
            return this;
        }

        public FullyConnectedLayer UseActivation(Activation activation)
        {
            Activation = activation;
            return this;
        }

        public void SetWeights(Matrix weights)
        {
            Weights = weights;
        }

        public void SetBias(Vector bias)
        {
            Bias = bias;
        }

        public void AutoInitWeightsBias()
        {
            Weights = emu.RandomMatrixGaussian(InputFeatures, NeuronCount);
            Bias = emu.RandomVectorGaussian(NeuronCount);
        }

        public Matrix Backward(Matrix outputError)
        {
            OutputError = outputError;
            ErrorBP();
            UpdateWeightsBias();
            return InputError;
        }

        private void ErrorBP()
        {
            LinearError = Activation.Backward(ForwardOutput, OutputError);
            InputError = LinearError * Weights.Transpose();
        }

        private void UpdateWeightsBias()
        {
        }

        public Matrix Forward(Matrix input)
        {
            ForwardInput = input;
            //Add Bias(a Vector) to Matrix row by row
            ForwardLinear = Bias + input * Weights;

            if (Activation != null)
                ForwardOutput = Activation.Forward(ForwardLinear);
            else
                ForwardOutput = ForwardLinear;

            return ForwardOutput;
        }
    }
}
