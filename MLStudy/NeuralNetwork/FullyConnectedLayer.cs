using MLStudy.Optimization;
using MLStudy.Regularizations;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class FullyConnectedLayer
    {
        public Matrix Weights { get; private set; }
        public Vector Bias { get; private set; }
        public int InputFeatures { get; private set; }
        public int NeuronCount { get; private set; }

        public Activation Activation { get; private set; }
        public Regularization Regularization { get; private set; }
        public GradientOptimizer Optimizer { get; private set; } = new NormalDescent();

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
            Weights = DataEmulator.Instance.RandomMatrixGaussian(InputFeatures, NeuronCount);
            Bias = DataEmulator.Instance.RandomVectorGaussian(NeuronCount);
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

        #region Activation

        public FullyConnectedLayer UseActivation(ActivationTypes activation, params int[] layers)
        {
            UseActivation(activation.ToString());
            return this;
        }

        public FullyConnectedLayer UseActivation(string activationName, params int[] layers)
        {
            Activation = Activation.Get(activationName);
            return this;
        }

        public FullyConnectedLayer UseActivation(Activation activation, params int[] layers)
        {
            Activation = activation;
            return this;
        }

        #endregion

        #region Regularization

        public FullyConnectedLayer UseRegularization(RegularTypes regularType, double weight)
        {
            return UseRegularization(regularType.ToString(), weight);
        }

        public FullyConnectedLayer UseRegularization(string regularType, double weight)
        {
            Regularization = Regularization.Get(regularType);
            if(Regularization != null)
                Regularization.Weight = weight;
            return this;
        }

        public FullyConnectedLayer UseLasso(double weight)
        {
            Regularization = new Lasso { Weight = weight };
            return this;
        }

        public FullyConnectedLayer UseRegularizationL1(double weight)
        {
            Regularization = new Lasso { Weight = weight };
            return this;
        }

        public FullyConnectedLayer UseRidge(double weight)
        {
            Regularization = new Ridge { Weight = weight };
            return this;
        }

        public FullyConnectedLayer UseRegularizationL2(double weight)
        {
            Regularization = new Ridge { Weight = weight };
            return this;
        }

        public FullyConnectedLayer UseRegularization(Regularization regular)
        {
            Regularization = regular;
            return this;
        }

        #endregion

        #region Optimization

        #endregion
    }
}
