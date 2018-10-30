/*
 * Description:Neural network model.
 * Author:Yunxiao An
 * Date:2018.10.26
 */

using MLStudy.Regularizations;
using System.Collections.Generic;
using System.Linq;

namespace MLStudy
{
    public class NeuralNetwork : ITrainable
    {
        public List<FullyConnectedLayer> HiddenLayers { get; private set; }
        public OutputLayer OutLayer { get; private set; }
        public int InputFeatures { get; private set; }

        public NeuralNetwork(int inputFeatures, params int[] hiddenLayers)
        {
            InputFeatures = inputFeatures;
            InitHiddenLayers(hiddenLayers);
        }

        private void InitHiddenLayers(int[] hiddenLayers)
        {
            HiddenLayers = new List<FullyConnectedLayer>();
            foreach (var count in hiddenLayers)
            {
                AddHiddenLayer(count);
            }
        }

        public void InitWeightsBias()
        {
            foreach (var layer in HiddenLayers)
            {
                layer.AutoInitWeightsBias();
            }
            OutLayer.AutoInitWeightsBias();
        }

        public void Test()
        {
            var testX = DataEmulator.Instance.RandomMatrixGaussian(1, InputFeatures);
            var testY = new Vector(1d);
            Forward(testX);
            Backward(testY);
        }

        public void Step(Matrix X, Vector y)
        {
            Forward(X);
            Backward(y);
        }

        private Matrix Forward(Matrix X)
        {
            Matrix input = X;

            for (int i = 0; i < HiddenLayers.Count; i++)
            {
                input = HiddenLayers[i].Forward(input);
            }

            return OutLayer.Forward(input);
        }

        private Matrix Backward(Vector y)
        {
            var outputError = OutLayer.Backward(y);

            for (int i = 0; i < HiddenLayers.Count; i++)
            {
                var l = HiddenLayers[HiddenLayers.Count - i - 1];
                outputError = l.Backward(outputError);
            }

            return outputError;
        }

        public double Loss(Matrix X, Vector y)
        {
            Forward(X);
            return OutLayer.GetLoss(y);
        }

        public void SetLearningRate(double learningRate, params int[] layers)
        {
            if (layers.Length == 0)
            {
                foreach (var layer in HiddenLayers)
                {
                    layer.LearningRate = learningRate;
                }
            }
            else
            {
                foreach (var index in layers)
                {
                    HiddenLayers[index].LearningRate = learningRate;
                }
            }

            if (OutLayer != null)
                OutLayer.LearningRate = learningRate;
        }

        #region Layers

        public NeuralNetwork AddHiddenLayer(int neuronCount)
        {
            var input = InputFeatures;
            if (HiddenLayers.Count != 0)
                input = HiddenLayers.Last().NeuronCount;

            var layer = new FullyConnectedLayer(input, neuronCount);
            AddHiddenLayer(layer);
            return this;
        }

        public NeuralNetwork AddHiddenLayer(int neuronCount, ActivationTypes activationType, WeightDecayTypes regularType)
        {
            var input = InputFeatures;
            if (HiddenLayers.Count != 0)
                input = HiddenLayers.Last().NeuronCount;

            var layer = new FullyConnectedLayer(input, neuronCount);
            AddHiddenLayer(layer);
            var index = HiddenLayers.IndexOf(layer);
            UseActivation(activationType, index);
            UseWeightDecay(regularType, index);
            return this;
        }

        public NeuralNetwork AddHiddenLayer(FullyConnectedLayer layer)
        {
            HiddenLayers.Add(layer);
            if (OutLayer != null)
                OutLayer.InputFeatures = layer.NeuronCount;
            return this;
        }

        public NeuralNetwork UseOutputLayer(OutputLayer outLayer)
        {
            OutLayer = outLayer;
            return this;
        }

        public NeuralNetwork UseLinearRegressionOutLayer()
        {
            var input = InputFeatures;
            if (HiddenLayers.Count != 0)
                input = HiddenLayers.Last().NeuronCount;
                
            OutLayer = new LinearRegressionOut(input);
            return this;
        }

        public NeuralNetwork UseLogisticRegressionOutLayer()
        {
            var input = InputFeatures;
            if (HiddenLayers.Count != 0)
                input = HiddenLayers.Last().NeuronCount;

            OutLayer = new LogisticRegressionOut(input);
            return this;
        }

        public NeuralNetwork UseSoftmaxOutLayer(int categoryCount)
        {
            var input = InputFeatures;
            if (HiddenLayers.Count != 0)
                input = HiddenLayers.Last().NeuronCount;

            OutLayer = new SoftmaxOut(input, categoryCount);
            return this;
        }

        #endregion

        #region Activation

        public NeuralNetwork UseActivation(ActivationTypes activation, params int[] layers)
        {
            UseActivation(activation.ToString(), layers);
            return this;
        }

        public NeuralNetwork UseActivation(string activationName, params int[] layers)
        {
            foreach (var index in layers)
            {
                var act = Activation.Get(activationName);
                UseActivation(act, index);
            }
            return this;
        }

        public NeuralNetwork UseActivation(Activation activation, int layerIndex)
        {
            HiddenLayers[layerIndex].UseActivation(activation);
            return this;
        }

        #endregion

        #region Weight Decay

        public NeuralNetwork UseWeightDecay(WeightDecayTypes regularType, double strength, params int[] layers)
        {
            return UseWeightDecay(regularType.ToString(), strength);
        }

        public NeuralNetwork UseWeightDecay(string regularType, double strength, params int[] layers)
        {
            if (layers.Length == 0)
            {
                foreach (var layer in HiddenLayers)
                {
                    var decay = WeightDecay.Get(regularType);
                    if (decay != null)
                        decay.Strength = strength;
                    layer.UseWeightDecay(decay);
                }

                var outDecay = WeightDecay.Get(regularType);
                if (outDecay != null)
                    outDecay.Strength = strength;
                OutLayer.UseWeightDecay(outDecay);
            }
            else
            {
                foreach (var index in layers)
                {
                    var reg = WeightDecay.Get(regularType);
                    if (reg != null)
                        reg.Strength = strength;
                    UseWeightDecay(reg, index);
                }
            }

            return this;
        }

        public NeuralNetwork UseLasso(double strength, params int[] layers)
        {
            return UseWeightDecay("L1", strength, layers);
        }

        public NeuralNetwork UseWeightDecayL1(double strength, params int[] layers)
        {
            return UseWeightDecay("L1", strength, layers);
        }

        public NeuralNetwork UseRidge(double strength, params int[] layers)
        {
            return UseWeightDecay("L2", strength, layers);
        }

        public NeuralNetwork UseWeightDecayL2(double strength, params int[] layers)
        {
            return UseWeightDecay("L2", strength, layers);
        }

        public NeuralNetwork UseWeightDecay(WeightDecay decay, int layerIndex)
        {
            if (layerIndex < 0)
                OutLayer.UseWeightDecay(decay);
            else
                HiddenLayers[layerIndex].UseWeightDecay(decay);

            return this;
        }

        #endregion

        #region Optimization

        public NeuralNetwork UseOptimizer(GradientOptimizer optimizer, int layerIndex)
        {
            if (layerIndex < 0)
                OutLayer.UseOptimizer(optimizer);
            else
                HiddenLayers[layerIndex].UseOptimizer(optimizer);

            return this;
        }

        public NeuralNetwork UseMomentum()
        {
            return this;
        }

        public NeuralNetwork UseAdam()
        {
            return this;
        }

        public NeuralNetwork UseBatchNorm()
        {
            return this;
        }

        #endregion
    }
}
