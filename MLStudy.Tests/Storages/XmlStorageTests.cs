using MLStudy.Abstraction;
using MLStudy.Deep;
using MLStudy.PreProcessing;
using MLStudy.Storages;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Xml;
using Xunit;

namespace MLStudy.Tests.Storages
{
    public class XmlStorageTests
    {
        [Fact]
        public void LossStorageTest()
        {
            var loss = new CrossEntropy();
            var xml = new XmlDocument();
            var el =XmlStorage.SaveToEl(xml, loss);
            var test = XmlStorage.LoadFromNode<CrossEntropy>(el);

            Assert.True(test is CrossEntropy);
        }

        [Fact]
        public void LossStorageTest2()
        {
            var loss = new MeanSquareError();
            var xml = new XmlDocument();
            var el = XmlStorage.SaveToEl(xml, loss);
            var test = XmlStorage.LoadFromNode<MeanSquareError>(el);

            Assert.True(test is MeanSquareError);
        }

        [Fact]
        public void TensorStorageTest()
        {
            var tensor = new TensorOld(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 }, 1, 1, 2, 4);
            var xml = new XmlDocument();
            var el = XmlStorage.SaveToEl(xml, tensor);
            var test = XmlStorage.LoadFromNode<TensorOld>(el);
            Assert.Equal(tensor, test);
        }

        [Fact]
        public void FullLayerStorageTest()
        {
            var weights = new TensorOld(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 }, 4, 2);
            var bias = new TensorOld(new double[] { 3, 4 }, 1, 2);
            var xml = new XmlDocument();

            var layer = new FullLayer(2);
            layer.SetWeights(weights);
            layer.SetBias(bias);

            var el = XmlStorage.SaveToEl(xml, layer);
            var test = XmlStorage.LoadFromNode<ILayer>(el);
            Assert.True(test is FullLayer);

            var full = test as FullLayer;
            Assert.Equal(2, full.UnitCount);
            Assert.Equal(weights, full.Weights);
            Assert.Equal(bias, full.Bias);
        }

        [Fact]
        public void MaxPoolingStorageTest()
        {
            var max = new MaxPooling(2, 2);
            var doc = new XmlDocument();
            var el = XmlStorage.SaveToEl(doc, max);
            var test = XmlStorage.LoadFromNode<MaxPooling>(el);

            Assert.True(test is MaxPooling);
            Assert.Equal(max.Rows, test.Rows);
            Assert.Equal(max.Columns, test.Columns);
            Assert.Equal(max.RowStride, test.RowStride);
            Assert.Equal(max.ColumnStride, test.ColumnStride);
        }

        [Fact]
        public void MeanPoolingStorageTest()
        {
            var mean = new MeanPooling(2, 2);
            var doc = new XmlDocument();
            var el = XmlStorage.SaveToEl(doc, mean);
            var test = XmlStorage.LoadFromNode<MeanPooling>(el);

            Assert.True(test is MeanPooling);
            Assert.Equal(mean.Rows, test.Rows);
            Assert.Equal(mean.Columns, test.Columns);
            Assert.Equal(mean.RowStride, test.RowStride);
            Assert.Equal(mean.ColumnStride, test.ColumnStride);
        }

        [Fact]
        public void ConvLayerStorageTest()
        {
            var doc = new XmlDocument();
            var layer = new ConvLayer(3, 2,1,1);
            layer.SetFilters(new TensorOld(
                new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, 3, 1, 2, 2));
            layer.SetBias(new TensorOld(new double[] { 1, 2, 3 }));
            var el = XmlStorage.SaveToEl(doc, layer);
            var test = XmlStorage.LoadFromNode<ConvLayer>(el);

            Assert.True(test is ConvLayer);
            Assert.Equal(layer.PaddingValue, test.PaddingValue);
            Assert.Equal(layer.FilterRows, test.FilterRows);
            Assert.Equal(layer.FilterColumns, test.FilterColumns);
            Assert.Equal(layer.RowStride, test.RowStride);
            Assert.Equal(layer.ColumnStride, test.ColumnStride);
            Assert.Equal(layer.Filters, test.Filters);
            Assert.Equal(layer.Bias, test.Bias);
        }

        [Fact]
        public void NeuralNetworkTest()
        {
            var model = new NeuralNetwork()
                .AddFullLayer(10)
                .AddSigmoid()
                .AddFullLayer(6)
                .AddSigmoid()
                .AddFullLayer(3)
                .AddSoftmax()
                .UseCrossEntropyLoss()
                .UseGradientDescent(0.01);

            Storage.Save(model, "nn.xml");

            var test = Storage.Load<NeuralNetwork>("nn.xml");

            Assert.Equal(model.PredictLayers.Count, test.PredictLayers.Count);
            Assert.Equal(model.TrainingLayers.Count, test.TrainingLayers.Count);
            Assert.Null(test.Regularizer);
            Assert.IsType<GradientDescent>(test.Optimizer);
            Assert.IsType<CrossEntropy>(test.LossFunction);
        }

        [Fact]
        public void TrainerTest()
        {
            var model = new NeuralNetwork()
                .AddFullLayer(10)
                .AddSigmoid()
                .AddFullLayer(6)
                .AddSigmoid()
                .AddFullLayer(3)
                .AddSoftmax()
                .UseCrossEntropyLoss()
                .UseAdam();

            var trainer = new Trainer(model, 64, 10)
            {
                Mission = "MNIST",
                LabelCodec = new OneHotCodec(new List<string> { "a", "b", "c" }),
            };

            var doc = new XmlDocument();
            Storage.Save(trainer, "trainer.xml");

            var test = Storage.Load<Trainer>("trainer.xml");
        }
    }
}
