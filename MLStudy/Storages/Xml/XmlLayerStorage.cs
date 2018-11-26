using MLStudy.Abstraction;
using MLStudy.Deep;
using MLStudy.Storages.Xml;
using System;
using System.Collections.Generic;
using System.Text;
using System.Xml;

namespace MLStudy.Storages
{
    public class XmlReLUStorage : XmlStorageBase<ReLU> { }
    public class XmlTanhStorage : XmlStorageBase<Tanh> { }
    public class XmlSigmoidStorage : XmlStorageBase<Sigmoid> { }
    public class XmlSoftmaxStorage : XmlStorageBase<Softmax> { }

    public class XmlFullLayerStorage : XmlStorageBase<FullLayer>
    {
        public override XmlElement Save(XmlDocument doc, object o)
        {
            if (!(o is FullLayer layer))
                throw new Exception("layer must be FullLayer!");

            var el = GetRootElement(doc);

            XmlStorage.AddChild(el, "UnitCount", layer.UnitCount);
            XmlStorage.AddObjectChild(el, "Weights", layer.Weights);
            XmlStorage.AddObjectChild(el, "Bias", layer.Bias);

            return el;
        }

        public override object Load(XmlNode node)
        {
            if (node == null) return null;

            if (!node.HasChildNodes)
                return null;

            var unitCount = XmlStorage.GetIntValue(node, "UnitCount");
            var result = (FullLayer)Activator.CreateInstance(typeof(FullLayer), unitCount);

            var weights = XmlStorage.GetObjectValue<Tensor>(node, "Weights");
            if (weights != null)
                result.SetWeights(weights);

            var bias = XmlStorage.GetObjectValue<Tensor>(node, "Bias");
            if (bias != null)
                result.SetBias(bias);

            return result;
        }
    }

    public class XmlMaxPoolingStorage : XmlStorageBase<MaxPooling>
    {
        public override object Load(XmlNode node)
        {
            var rows = XmlStorage.GetIntValue(node, "Rows");
            var columns = XmlStorage.GetIntValue(node, "Columns");
            var rowStride = XmlStorage.GetIntValue(node, "RowStride");
            var columnStride = XmlStorage.GetIntValue(node, "ColumnStride");

            return Activator.CreateInstance(typeof(MaxPooling), rows, columns, rowStride, columnStride);
        }

        public override XmlElement Save(XmlDocument doc, object o)
        {
            if (!(o is MaxPooling layer))
                throw new Exception("layer must be MaxPooling!");

            return SavePoolingLayer(Name, doc, layer);
        }

        internal static XmlElement SavePoolingLayer(string type, XmlDocument doc, PoolingLayer layer)
        {
            var el = doc.CreateElement(type);

            XmlStorage.AddChild(el, "Rows", layer.Rows.ToString());
            XmlStorage.AddChild(el, "Columns", layer.Columns.ToString());
            XmlStorage.AddChild(el, "RowStride", layer.RowStride.ToString());
            XmlStorage.AddChild(el, "ColumnStride", layer.ColumnStride.ToString());

            return el;
        }
    }

    public class XmlMeanPoolingStorage : XmlStorageBase<MeanPooling>
    {
        public override object Load(XmlNode node)
        {
            var rows = XmlStorage.GetIntValue(node, "Rows");
            var columns = XmlStorage.GetIntValue(node, "Columns");
            var rowStride = XmlStorage.GetIntValue(node, "RowStride");
            var columnStride = XmlStorage.GetIntValue(node, "ColumnStride");

            return Activator.CreateInstance(typeof(MeanPooling), rows, columns, rowStride, columnStride);
        }

        public override XmlElement Save(XmlDocument doc, object o)
        {
            if (!(o is MeanPooling layer))
                throw new Exception("layer must be MeanPooling!");

            return XmlMaxPoolingStorage.SavePoolingLayer(Name, doc, layer);
        }
    }

    public class XmlConvLayerStorage : XmlStorageBase<ConvLayer>
    {
        public override object Load(XmlNode node)
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

            var filters = XmlStorage.GetObjectValue<Tensor>(node, "Filters");
            if (filters != null)
                result.SetFilters(filters);

            var bias = XmlStorage.GetObjectValue<Tensor>(node, "Bias");
            if (bias != null)
                result.SetBias(bias);

            return result;
        }

        public override XmlElement Save(XmlDocument doc, object o)
        {
            if (!(o is ConvLayer layer))
                throw new Exception("layer must be ConvLayer!");

            var el = GetRootElement(doc);

            XmlStorage.AddChild(el, "FilterCount", layer.FilterCount.ToString());
            XmlStorage.AddChild(el, "FilterRows", layer.FilterRows.ToString());
            XmlStorage.AddChild(el, "FilterColumns", layer.FilterColumns.ToString());
            XmlStorage.AddChild(el, "RowStride", layer.RowStride.ToString());
            XmlStorage.AddChild(el, "ColumnStride", layer.ColumnStride.ToString());
            XmlStorage.AddChild(el, "RowPadding", layer.RowPadding.ToString());
            XmlStorage.AddChild(el, "ColumnPadding", layer.ColumnPadding.ToString());
            XmlStorage.AddChild(el, "PaddingValue", layer.PaddingValue.ToString());

            XmlStorage.AddObjectChild(el, "Filters", layer.Filters);
            XmlStorage.AddObjectChild(el, "Bias", layer.Bias);

            return el;
        }
    }

}
