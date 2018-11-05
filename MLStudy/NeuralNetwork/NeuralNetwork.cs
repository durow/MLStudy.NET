/*
 * Description:Neural network model.
 * Author:Yunxiao An
 * Date:2018.10.26
 */

using MLStudy.Abstraction;
using MLStudy.Regularizations;
using System.Collections.Generic;
using System.Linq;

namespace MLStudy
{
    public class NeuralNetwork : ITrainable
    {
        public List<CNNLayer> CNNLayers { get; private set; }
        public FlattenLayer FlattenLayer { get; private set; } = new FlattenLayer();
        public List<FullyConnectedLayer> FCLayers { get; private set; }
        public OutputLayer OutLayer { get; private set; }
        public int InputFeatures { get; private set; }

        public int InputRows { get; private set; }
        public int InputColumns { get; private set; }
        public int InputDepth { get; private set; }

        public NeuralNetwork()
        {
            CNNLayers = new List<CNNLayer>();
            FCLayers = new List<FullyConnectedLayer>();
        }

        public NeuralNetwork(int fcInputFeatures, params int[] FCLayers)
        {
            InputFeatures = fcInputFeatures;
            InitFCLayers(FCLayers);
        }

        private void InitFCLayers(int[] fcLayers)
        {
            FCLayers = new List<FullyConnectedLayer>();
            foreach (var count in fcLayers)
            {
                AddHiddenLayer(count);
            }
        }

        public void InitWeightsBias()
        {
            foreach (var layer in FCLayers)
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

        public Vector Predict(Matrix X)
        {
            Forward(X);
            return OutLayer.GetPredict();
        }

        public void Step(Matrix X, Vector y)
        {
            Forward(X);
            Backward(y);
        }

        public Matrix Forward(Matrix X)
        {
            Matrix input = X;

            for (int i = 0; i < FCLayers.Count; i++)
            {
                input = FCLayers[i].Forward(input);
            }

            return OutLayer.Forward(input);
        }

        public Matrix Forward(Tensor X)
        {
            var input = X;

            for (int i = 0; i < CNNLayers.Count; i++)
            {
                input = CNNLayers[i].Forward(input);
            }

            var fcInput = FlattenLayer.FlattenToMatrix(input);
            return Forward(fcInput);
        }

        private Matrix Backward(Vector y)
        {
            var outputError = OutLayer.Backward(y);

            for (int i = 0; i < FCLayers.Count; i++)
            {
                var l = FCLayers[FCLayers.Count - i - 1];
                outputError = l.Backward(outputError);
            }

            return outputError;
        }

        public double Loss(Matrix X, Vector y)
        {
            Forward(X);
            return OutLayer.GetLoss(y);
        }

        public double GetLastError(Vector y)
        {
            return OutLayer.GetError(y);
        }

        public double GetLastLoss(Vector y)
        {
            return OutLayer.GetLoss(y);
        }

        public void SetLearningRate(double learningRate, params int[] layers)
        {
            if (layers.Length == 0)
            {
                foreach (var layer in FCLayers)
                {
                    layer.LearningRate = learningRate;
                }
            }
            else
            {
                foreach (var index in layers)
                {
                    FCLayers[index].LearningRate = learningRate;
                }
            }

            if (OutLayer != null)
                OutLayer.LearningRate = learningRate;
        }

        #region Layers

        public NeuralNetwork AddHiddenLayer(int neuronCount)
        {
            var input = InputFeatures;
            if (FCLayers.Count != 0)
                input = FCLayers.Last().NeuronCount;

            var layer = new FullyConnectedLayer(input, neuronCount);
            AddFullyConnectedLayer(layer);
            return this;
        }

        public NeuralNetwork AddConvolutionalLayers(params ConvolutionalLayer[] layers)
        {
            foreach (var layer in layers)
            {
                CNNLayers.Add(layer);
            }
            return this;
        }

        public NeuralNetwork AddConvolutionalLayer(int filterRows, int filterColumns, int filterCount, 
            int filterDepth = 0, int padding = 0, int stride = 1)
        {
            if(filterDepth <= 0)
            {
                if (CNNLayers.Count == 0)
                    throw new System.Exception("must specify the depth of the first ConvolutionalLayer!");
                var last = CNNLayers.Last(l => l is ConvolutionalLayer);
                if(last == null)
                    throw new System.Exception("must specify the depth of the first ConvolutionalLayer!");

                filterDepth = ((ConvolutionalLayer)last).FilterCount;
            }

            var layer = new ConvolutionalLayer(filterDepth, filterRows, filterColumns, filterCount)
            {
                Padding = padding,
                Stride = stride
            };
            return AddConvolutionalLayers(layer);
        }

        public NeuralNetwork AddConvolutionalLayer(int filterRows, int filterColumns, int filterCount, int layerCount,
            int firstLayerDepth = 0, int padding = 0, int stride = 1)
        {
            if (firstLayerDepth <= 0)
            {
                if (CNNLayers.Count == 0)
                    throw new System.Exception("must specify the depth of the first ConvolutionalLayer!");
                var last = CNNLayers.Last(l => l is ConvolutionalLayer);
                if (last == null)
                    throw new System.Exception("must specify the depth of the first ConvolutionalLayer!");

                firstLayerDepth = ((ConvolutionalLayer)last).FilterCount;
            }

            var layer = new ConvolutionalLayer(firstLayerDepth, filterRows, filterColumns, filterCount)
            {
                Padding = padding,
                Stride = stride
            };
            AddConvolutionalLayers(layer);

            for (int i = 1; i < layerCount; i++)
            {
                layer = new ConvolutionalLayer(layer.FilterCount, filterRows, filterColumns, filterCount);
                AddConvolutionalLayers(layer);
            }
            return this;
        }

        public NeuralNetwork AddPoolingLayer(PoolingLayer layer)
        {
            CNNLayers.Add(layer);
            return this;
        }

        public NeuralNetwork AddPoolingLayer(int rows = 2, int columns = 2, int stride = 2, PoolingType type = PoolingType.Max)
        {
            var layer = new PoolingLayer(rows, columns, stride, type);
            return AddPoolingLayer(layer);
        }

        public NeuralNetwork AddFullyConnectedLayer(int neuronCount, ActivationTypes activationType, WeightDecayTypes regularType)
        {
            var input = InputFeatures;
            if (FCLayers.Count != 0)
                input = FCLayers.Last().NeuronCount;

            var layer = new FullyConnectedLayer(input, neuronCount);
            AddFullyConnectedLayer(layer);
            var index = FCLayers.IndexOf(layer);
            UseActivation(activationType, index);
            UseWeightDecay(regularType, index);
            return this;
        }

        public NeuralNetwork AddFullyConnectedLayer(FullyConnectedLayer layer)
        {
            FCLayers.Add(layer);
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
            if (FCLayers.Count != 0)
                input = FCLayers.Last().NeuronCount;
                
            OutLayer = new LinearRegressionOut(input);
            return this;
        }

        public NeuralNetwork UseLogisticRegressionOutLayer()
        {
            var input = InputFeatures;
            if (FCLayers.Count != 0)
                input = FCLayers.Last().NeuronCount;

            OutLayer = new LogisticRegressionOut(input);
            return this;
        }

        public NeuralNetwork UseSoftmaxOutLayer(int categoryCount)
        {
            var input = InputFeatures;
            if (FCLayers.Count != 0)
                input = FCLayers.Last().NeuronCount;

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
            FCLayers[layerIndex].UseActivation(activation);
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
                foreach (var layer in FCLayers)
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
                FCLayers[layerIndex].UseWeightDecay(decay);

            return this;
        }

        #endregion

        #region Optimization

        public NeuralNetwork UseOptimizer(GradientOptimizer optimizer, int layerIndex)
        {
            if (layerIndex < 0)
                OutLayer.UseOptimizer(optimizer);
            else
                FCLayers[layerIndex].UseOptimizer(optimizer);

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
