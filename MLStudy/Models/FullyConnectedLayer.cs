using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class FullyConnectedLayer
    {
        private DataEmulator emu = new DataEmulator();
        private readonly Activation activation = new Activation(Activations.ReLU);


        public Matrix Weights { get; private set; }
        public Vector Bias { get; private set; }
        public int NeuronCount { get; private set; }
        public int InputFeatures { get; private set; }
        public double LearningRate { get; set; }
        public Vector this[int index]
        {
            get
            {
                return Weights.GetColumn(index);
            }
        }
        public Activations Activation
        {
            get
            {
                return activation.ActivationType;
            }
            set
            {
                activation.ActivationType = value;
            }
        }
        public Matrix LastForward { get; private set; }
        public Matrix LastBackward { get; private set; }

        public FullyConnectedLayer(int inputFeatures, int neuronCount)
        {
            InputFeatures = inputFeatures;
            NeuronCount = neuronCount;
            AutoInitWeightsBias();
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

        public void Backward()
        {
            throw new NotImplementedException();
        }

        private void ErrorBP()
        { }

        private void UpdateWeightsBias()
        { }

        public Matrix Forward(Matrix input)
        {
            LastForward = Bias + input * Weights;
            return LastForward;
        }
    }
}
