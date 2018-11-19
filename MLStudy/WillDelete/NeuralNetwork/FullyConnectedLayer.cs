using MLStudy.Activations;
using MLStudy.Optimization;
using MLStudy.Regularizations;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class FullyConnectedLayer
    {
        public double LearningRate
        {
            get
            {
                return Optimizer.LearningRate;
            }
            set
            {
                Optimizer.LearningRate = value;
            }
        }
        public Matrix Weights { get; set; }
        public Vector Bias { get; set; }
        public int InputFeatures { get; private set; }
        public int NeuronCount { get; private set; }

        public Activation Activation { get; private set; } = new ReLU();
        public WeightDecay WeightDecay { get; private set; }
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

        public FullyConnectedLayer UseActivation(Activation activation)
        {
            Activation = activation;
            return this;
        }

        public FullyConnectedLayer UseWeightDecay(WeightDecay decay)
        {
            WeightDecay = decay;
            return this;
        }

        public FullyConnectedLayer UseOptimizer(GradientOptimizer optimizer)
        {
            Optimizer = optimizer;
            return this;
        }

        public FullyConnectedLayer UseMomentum()
        {
            return this;
        }

        public FullyConnectedLayer UseAdam()
        {
            return this;
        }

        public FullyConnectedLayer UseDropOut()
        {
            return this;
        }

        public FullyConnectedLayer UseBatchNorm()
        {
            return this;
        }


        #region Private Methods

        private void ErrorBP()
        {
            LinearError = Activation.Backward(ForwardOutput, OutputError);
            InputError = LinearError * Weights.Transpose();
        }

        private void UpdateWeightsBias()
        {
            var v = new Vector(LinearError.Rows, 1);
            var gradientBias = (v * LinearError).ToVector() / LinearError.Rows;
            var gradientWeights = ForwardInput.Transpose() * LinearError / LinearError.Rows;

            Weights = Optimizer.GradientDescent(Weights, gradientWeights);
            Bias = Optimizer.GradientDescent(Bias, gradientBias);
        }

        #endregion
    }
}
