using MLStudy.Abstraction;
using MLStudy.Deep;
using MLStudy.Storages.Xml;
using System;
using System.Collections.Generic;
using System.Text;
using System.Xml;

namespace MLStudy.Storages
{
    public class XmlNetworkStorage : XmlStorageBase<NeuralNetwork>
    {
        public override XmlElement Save(XmlDocument doc, object o)
        {
            if (!(o is NeuralNetwork nn))
                throw new Exception("object must be NeuralNetwork");

            var el = GetRootElement(doc);

            XmlStorage.AddObjectChild(el, "LossFunction", nn.LossFunction);
            XmlStorage.AddObjectChild(el, "Optimizer", nn.Optimizer);
            XmlStorage.AddObjectChild(el, "Regularizer", nn.Regularizer);

            var layers = doc.CreateElement("Layers");
            foreach (var item in nn.TrainingLayers)
            {
                XmlStorage.AddObjectChild(layers, item);
            }
            XmlStorage.AddChild(el, layers);

            return el;
        }

        public override object Load(XmlNode node)
        {
            var result = (NeuralNetwork)Activator.CreateInstance(Type);

            var loss = XmlStorage.GetObjectValue<LossFunction>(node, "LossFunction");
            result.UseLossFunction(loss);

            var optimizer = XmlStorage.GetObjectValue<IOptimizer>(node, "Optimizer");
            result.UseOptimizer(optimizer);

            var regularizer = XmlStorage.GetObjectValue<IRegularizer>(node, "Regularizer");
            result.UseRegularizer(regularizer);

            var nodeList = node.SelectSingleNode("Layers").ChildNodes;
            foreach (XmlNode n in nodeList)
            {
                var layer = XmlStorage.LoadFromNode<ILayer>(n);
                result.AddLayer(layer);
            }

            return result;
        }
    }
}
