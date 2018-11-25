using MLStudy.Abstraction;
using MLStudy.Deep;
using System;
using System.Collections.Generic;
using System.Text;
using System.Xml;

namespace MLStudy.Storage
{
    public class XmlLayerStorage
    {
        public static Dictionary<Type, Func<XmlDocument, ILayer, XmlElement>> LayerSaver = new Dictionary<Type, Func<XmlDocument, ILayer, XmlElement>>();

        public static Dictionary<string, Func<XmlNode, ILayer>> LayerLoader = new Dictionary<string, Func<XmlNode, ILayer>>();

        static XmlLayerStorage()
        {
            AddEmpty<ReLU>();
            AddEmpty<Sigmoid>();
            AddEmpty<Tanh>();
            AddEmpty<Softmax>();

            Add<FullLayer>(SaveFullLayer, LoadFullLayer);
            Add<MaxPooling>(SaveMaxPoolingLayer, LoadPoolingLayer);
            Add<MeanPooling>(SaveMeanPoolingLayer, LoadPoolingLayer);
        }

        public XmlElement Save(XmlDocument doc, ILayer layer)
        {
            var type = layer.GetType();
            if (!LayerSaver.ContainsKey(type))
                throw new Exception($"can't find {type.Name}'s saver!");

            return LayerSaver[type](doc, layer);
        }

        public ILayer Load(XmlNode node)
        {
            var type = node.Name;
            if (!LayerLoader.ContainsKey(type))
                throw new Exception($"can't find {type}'s loader!");

            return LayerLoader[type](node);
        }

        public static void Add<T>(Func<XmlDocument, ILayer, XmlElement> saver, Func<XmlNode, ILayer> loader)
        {
            LayerSaver[typeof(T)] = saver;
            LayerLoader[typeof(T).Name] = loader;
        }

        public static void AddEmpty<T>() where T : ILayer
        {
            var type = typeof(T);
            LayerSaver.Add(type, (d,o) => XmlLossStorage.SaveEmptyObject<T>(d));
            LayerLoader.Add(type.Name, e => XmlLossStorage.CreateEmptyObject<T>());
        }

        public static XmlElement SaveFullLayer(XmlDocument doc, ILayer layer)
        {
            var fullLayer = layer as FullLayer;
            if (fullLayer == null)
                throw new Exception("layer must be FullLayer!");

            var el = doc.CreateElement(typeof(FullLayer).Name);

            var unitCount = doc.CreateElement("UnitCount");
            unitCount.InnerText = fullLayer.UnitCount.ToString();

            var weights = doc.CreateElement("Weights");
            weights.AppendChild(XmlTensorStorage.Save(doc, fullLayer.Weights));

            var bias = doc.CreateElement("Bias");
            bias.AppendChild(XmlTensorStorage.Save(doc, fullLayer.Bias));
            
            el.AppendChild(unitCount);
            el.AppendChild(weights);
            el.AppendChild(bias);

            return el;
        }

        public static ILayer LoadFullLayer(XmlNode el)
        {
            var unitCountText = el.SelectSingleNode("UnitCount").InnerText;
            var unitCount = int.Parse(unitCountText);

            var result = (FullLayer)Activator.CreateInstance(typeof(FullLayer), unitCount);

            var weightsEl = el.SelectSingleNode("Weights/Tensor");
            var weights = XmlTensorStorage.Load(weightsEl);
            if (weights != null)
                result.SetWeights(weights);

            var biasEl = el.SelectSingleNode("Bias/Tensor");
            var bias = XmlTensorStorage.Load(biasEl);
            if (bias != null)
                result.SetBias(bias);

            return result;
        }

        public static XmlElement SaveMaxPoolingLayer(XmlDocument doc, ILayer layer)
        {
            var maxLayer = layer as MaxPooling;
            if (maxLayer == null)
                throw new Exception("layer must be MaxPooling!");

            return SavePoolingLayer("MaxPooling", doc, maxLayer);
        }

        public static XmlElement SaveMeanPoolingLayer(XmlDocument doc, ILayer layer)
        {
            var maxLayer = layer as MeanPooling;
            if (maxLayer == null)
                throw new Exception("layer must be MeanPooling!");

            return SavePoolingLayer("MeanPooling", doc, maxLayer);
        }

        public static ILayer LoadPoolingLayer(XmlNode node)
        {
            var rowsEl = node.SelectSingleNode("Rows");
            var rows = int.Parse(rowsEl.InnerText);

            var columnsEl = node.SelectSingleNode("Columns");
            var columns = int.Parse(columnsEl.InnerText);

            var rowStrideEl = node.SelectSingleNode("RowStride");
            var rowStride = int.Parse(rowStrideEl.InnerText);

            var columnStrideEl = node.SelectSingleNode("ColumnStride");
            var columnStride = int.Parse(columnStrideEl.InnerText);

            if(node.Name == "MaxPooling")
                return (MaxPooling)Activator.CreateInstance(typeof(MaxPooling), rows, columns, rowStride, columnStride);
            if (node.Name == "MeanPooling")
                return (MeanPooling)Activator.CreateInstance(typeof(MeanPooling), rows, columns, rowStride, columnStride);

            throw new Exception($"Unknown type {node.Name}!");
        }

        private static XmlElement SavePoolingLayer(string type, XmlDocument doc, PoolingLayer layer)
        {
            var el = doc.CreateElement(type);

            var rows = doc.CreateElement("Rows");
            rows.InnerText = layer.Rows.ToString();
            el.AppendChild(rows);

            var columns = doc.CreateElement("Columns");
            columns.InnerText = layer.Columns.ToString();
            el.AppendChild(columns);

            var rowStride = doc.CreateElement("RowStride");
            rowStride.InnerText = layer.RowStride.ToString();
            el.AppendChild(rowStride);

            var columnStride = doc.CreateElement("ColumnStride");
            columnStride.InnerText = layer.ColumnStride.ToString();
            el.AppendChild(columnStride);

            return el;
        }

        private static XmlElement SaveConvLayer(XmlDocument doc, ILayer layer)
        {
            throw new NotImplementedException();
        }
    }
}
