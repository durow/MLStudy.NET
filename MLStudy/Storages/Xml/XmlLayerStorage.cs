using MLStudy.Abstraction;
using MLStudy.Deep;
using System;
using System.Collections.Generic;
using System.Text;
using System.Xml;

namespace MLStudy.Storages
{
    public class XmlLayerStorage
    {
        private static Dictionary<Type, Func<XmlDocument, ILayer, XmlElement>> LayerSaver = new Dictionary<Type, Func<XmlDocument, ILayer, XmlElement>>();

        private static Dictionary<string, Func<XmlNode, ILayer>> LayerLoader = new Dictionary<string, Func<XmlNode, ILayer>>();

        static XmlLayerStorage()
        {
            AddEmpty<ReLU>();
            AddEmpty<Sigmoid>();
            AddEmpty<Tanh>();
            AddEmpty<Softmax>();

            Add<FullLayer>(SaveFullLayer, LoadFullLayer);
            Add<MaxPooling>(SaveMaxPoolingLayer, LoadPoolingLayer);
            Add<MeanPooling>(SaveMeanPoolingLayer, LoadPoolingLayer);
            Add<ConvLayer>(SaveConvLayer, LoadConvLayer);
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
            LayerSaver.Add(type, (d, o) => XmlStorage.SaveEmptyObject<T>(d));
            LayerLoader.Add(type.Name, e => XmlStorage.CreateEmptyObject<T>());
        }

        public static XmlElement SaveFullLayer(XmlDocument doc, ILayer layer)
        {
            if (!(layer is FullLayer fullLayer))
                throw new Exception("layer must be FullLayer!");

            var el = doc.CreateElement("FullLayer");

            if (layer == null)
                return el;

            XmlStorage.AddChild(el, "UnitCount", fullLayer.UnitCount);
            XmlStorage.AddTensorChild(el, "Weights", fullLayer.Weights);
            XmlStorage.AddTensorChild(el, "Bias", fullLayer.Bias);

            return el;
        }

        public static ILayer LoadFullLayer(XmlNode node)
        {
            if (node == null) return null;

            if (!node.HasChildNodes)
                return null;

            var unitCount = XmlStorage.GetIntValue(node, "UnitCount");
            var result = (FullLayer)Activator.CreateInstance(typeof(FullLayer), unitCount);

            var weights = XmlStorage.GetTensorValue(node, "Weights");
            if (weights != null)
                result.SetWeights(weights);

            var bias = XmlStorage.GetTensorValue(node, "Bias");
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
            var rows = XmlStorage.GetIntValue(node, "Rows");
            var columns = XmlStorage.GetIntValue(node, "Columns");
            var rowStride = XmlStorage.GetIntValue(node, "RowStride");
            var columnStride = XmlStorage.GetIntValue(node, "ColumnStride");

            if (node.Name == "MaxPooling")
                return (MaxPooling)Activator.CreateInstance(typeof(MaxPooling), rows, columns, rowStride, columnStride);
            if (node.Name == "MeanPooling")
                return (MeanPooling)Activator.CreateInstance(typeof(MeanPooling), rows, columns, rowStride, columnStride);

            throw new Exception($"Unknown type {node.Name}!");
        }

        private static XmlElement SavePoolingLayer(string type, XmlDocument doc, PoolingLayer layer)
        {
            var el = doc.CreateElement(type);

            XmlStorage.AddChild(el, "Rows", layer.Rows.ToString());
            XmlStorage.AddChild(el, "Columns", layer.Columns.ToString());
            XmlStorage.AddChild(el, "RowStride", layer.RowStride.ToString());
            XmlStorage.AddChild(el, "ColumnStride", layer.ColumnStride.ToString());

            return el;
        }

        private static XmlElement SaveConvLayer(XmlDocument doc, ILayer layer)
        {
            if (!(layer is ConvLayer cLayer))
                throw new Exception("layer must be ConvLayer!");

            var el = doc.CreateElement(layer.GetType().Name);

            XmlStorage.AddChild(el, "FilterCount", cLayer.FilterCount.ToString());
            XmlStorage.AddChild(el, "FilterRows", cLayer.FilterRows.ToString());
            XmlStorage.AddChild(el, "FilterColumns", cLayer.FilterColumns.ToString());
            XmlStorage.AddChild(el, "RowStride", cLayer.RowStride.ToString());
            XmlStorage.AddChild(el, "ColumnStride", cLayer.ColumnStride.ToString());
            XmlStorage.AddChild(el, "RowPadding", cLayer.RowPadding.ToString());
            XmlStorage.AddChild(el, "ColumnPadding", cLayer.ColumnPadding.ToString());
            XmlStorage.AddChild(el, "PaddingValue", cLayer.PaddingValue.ToString());

            XmlStorage.AddTensorChild(el, "Filters", cLayer.Filters);
            XmlStorage.AddTensorChild(el, "Bias", cLayer.Bias);

            return el;
        }

        private static ILayer LoadConvLayer(XmlNode node)
        {
            var count = XmlStorage.GetIntValue(node, "FilterCount");
            var rows = XmlStorage.GetIntValue(node, "FilterRows");
            var columns = XmlStorage.GetIntValue(node, "FilterColumns");
            var rowStride = XmlStorage.GetIntValue(node, "RowStride");
            var columnStride = XmlStorage.GetIntValue(node, "ColumnStride");
            var rowPadding = XmlStorage.GetIntValue(node, "RowPadding");
            var columnPadding = XmlStorage.GetIntValue(node, "ColumnPadding");
            var paddingValue = XmlStorage.GetDoubleValue(node, "PaddingValue");

            var result = (ConvLayer)Activator.CreateInstance(typeof(ConvLayer), count, rows, columns, rowStride, columnStride, rowPadding, columnPadding);
            result.PaddingValue = paddingValue;

            var filters = XmlStorage.GetTensorValue(node, "Filters");
            if (filters != null)
                result.SetFilters(filters);

            var bias = XmlStorage.GetTensorValue(node, "Bias");
            if (bias != null)
                result.SetBias(bias);

            return result;
        }
        
    }
}
