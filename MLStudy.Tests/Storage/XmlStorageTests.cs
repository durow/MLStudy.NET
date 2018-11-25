using MLStudy.Deep;
using MLStudy.Storage;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Xml;
using Xunit;

namespace MLStudy.Tests.Storage
{
    public class XmlStorageTests
    {
        [Fact]
        public void LossStorageTest()
        {
            var loss = new CrossEntropy();
            var xml = new XmlDocument();
            var el = XmlLossStorage.Save(xml, loss);
            xml.AppendChild(el);
            xml.Save("ce.xml");

            var str = File.ReadAllText("ce.xml");
            Assert.Equal(@"<CrossEntropy />", str);

            var result = XmlLossStorage.Load(el);
            Assert.IsType<CrossEntropy>(result);
        }

        [Fact]
        public void LossStorageTest2()
        {
            var loss = new MeanSquareError();
            var xml = new XmlDocument();
            var el = XmlLossStorage.Save(xml, loss);
            xml.AppendChild(el);
            xml.Save("mse.xml");

            var str = File.ReadAllText("mse.xml");
            Assert.Equal(@"<MeanSquareError />", str);

            var result = XmlLossStorage.Load(el);
            Assert.IsType<MeanSquareError>(result);
        }

        [Fact]
        public void TensorStorageTest()
        {
            var tensor = new Tensor(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 }, 1, 1, 2, 4);
            var xml = new XmlDocument();
            var el = XmlTensorStorage.Save(xml, tensor);
            xml.AppendChild(el);
            xml.Save("tensor.xml");

            var test = XmlTensorStorage.Load(el);
            Assert.Equal(tensor, test);
        }

        [Fact]
        public void FullLayerStorageTest()
        {
            var weights = new Tensor(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 }, 4, 2);
            var bias = new Tensor(new double[] { 3, 4 }, 1, 2);
            var xml = new XmlDocument();

            var layer = new FullLayer(2);
            layer.SetWeights(weights);
            layer.SetBias(bias);

            var el = XmlLayerStorage.SaveFullLayer(xml, layer);
            xml.AppendChild(el);
            xml.Save("fulllayer.xml");

            var test = XmlLayerStorage.LoadFullLayer(el) as FullLayer;
            Assert.Equal(2, layer.UnitCount);
            Assert.Equal(weights, test.Weights);
            Assert.Equal(bias, test.Bias);
        }
    }
}
