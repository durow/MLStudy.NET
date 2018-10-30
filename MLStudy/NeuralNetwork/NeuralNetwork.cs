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
        public double LearningRate { get; set; }
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

        public NeuralNetwork AddHiddenLayer(int neuronCount,
            ActivationTypes activationType = ActivationTypes.ReLU,
            WeightDecayType regularType = WeightDecayType.None)
        {
            var input = InputFeatures;
            if (HiddenLayers.Count != 0)
                input = HiddenLayers.Last().NeuronCount;

            var layer = new FullyConnectedLayer(input, neuronCount);
            HiddenLayers.Add(layer);
            var index = HiddenLayers.IndexOf(layer);
            UseActivation(activationType, index);
            UseWeightDecay(regularType, index);
            return this;
        }

        public NeuralNetwork AddHiddenLayer(FullyConnectedLayer layer)
        {
            HiddenLayers.Add(layer);
            return this;
        }

        public NeuralNetwork UseOutputLayer(OutputLayer outLayer)
        {
            OutLayer = outLayer;
            return this;
        }

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

        public NeuralNetwork UseWeightDecay(WeightDecayType regularType, double strength)
        {
            return UseWeightDecay(regularType.ToString(), strength);
        }

        public NeuralNetwork UseWeightDecay(string regularType, double strength)
        {
            var reg = WeightDecay.Get(regularType);
            if (reg != null)
                reg.Weight = strength;
            UseWeightDecay(reg);
            return this;
        }

        public NeuralNetwork UseLasso(double strength)
        {
            var reg = new Lasso { Weight = strength };
            UseWeightDecay(reg);
            return this;
        }

        public NeuralNetwork UseWeightDecayL1(double strength)
        {
            var reg = new Lasso { Weight = strength };
            UseWeightDecay(reg);
            return this;
        }

        public NeuralNetwork UseRidge(double strength)
        {
            var reg = new Ridge { Weight = strength };
            UseWeightDecay(reg);
            return this;
        }

        public NeuralNetwork UseWeightDecayL2(double strength)
        {
            var reg = new Ridge { Weight = strength };
            UseWeightDecay(reg);
            return this;
        }

        public NeuralNetwork UseWeightDecay(WeightDecay decay)
        {
            return this;
        }

        #endregion

        #region Optimization

        public NeuralNetwork UseMomentum()
        {
            return this;
        }

        public NeuralNetwork UseAdam()
        {
            return this;
        }

        #endregion
    }
}
