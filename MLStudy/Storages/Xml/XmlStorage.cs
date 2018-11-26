using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Xml;
using MLStudy.Abstraction;
using MLStudy.Deep;

namespace MLStudy.Storages
{
    public class XmlStorage : Storage
    {
        private static Dictionary<Type, Func<XmlDocument, object, XmlElement>> savers = new Dictionary<Type, Func<XmlDocument, object, XmlElement>>();
        private static Dictionary<string, Func<XmlNode, object>> loaders = new Dictionary<string, Func<XmlNode, object>>();

        static XmlStorage()
        {
            var ass = Assembly.GetExecutingAssembly();
            var types = ass.GetTypes().Where(t => t.IsClass);
            foreach (var item in types)
            {
                var constructures = item.GetConstructors();
                foreach (var con in constructures)
                {
                    if (con.GetParameters().Length == 0)
                    {
                        AddEmptyStorage(item);
                        break;
                    }
                }
            }

            AddStorage<Tensor>(SaveTensor, LoadTensor);
            AddStorage<FullLayer>(SaveFullLayer, LoadFullLayer);
            AddStorage<MaxPooling>(SaveMaxPoolingLayer, LoadPoolingLayer);
            AddStorage<MeanPooling>(SaveMeanPoolingLayer, LoadPoolingLayer);
            AddStorage<ConvLayer>(SaveConvLayer, LoadConvLayer);
        }

        public override void SaveToFile(object o, string filename)
        {
            var type = o.GetType();
            if (!savers.ContainsKey(type))
                throw new Exception($"can't find {type}'s saver!");

            var doc = new XmlDocument();
            var dec = doc.CreateXmlDeclaration("1.0", "utf-8", null);
            doc.AppendChild(dec);

            var root = savers[type](doc, o);
            doc.AppendChild(root);
            doc.Save(filename);
        }

        public XmlElement SaveToEl(XmlDocument doc, object o)
        {
            var type = o.GetType();
            if (!savers.ContainsKey(type))
                throw new Exception($"can't find {type}'s saver!");

            return savers[type](doc, o);
        }

        public override T LoadFromFile<T>(string filename)
        {
            var doc = new XmlDocument();
            doc.Load(filename);
            var root = doc.FirstChild;
            return LoadFromNode<T>(root);
        }

        public T LoadFromNode<T>(XmlNode node)
        {
            var type = node.Name;
            if (!loaders.ContainsKey(type))
                throw new Exception($"can't find {type}'s loader!");

            return (T)loaders[type](node);
        }

        public static void AddEmptyStorage<T>()
        {
            var type = typeof(T);
            AddEmptyStorage(type);
        }

        public static void AddEmptyStorage(Type type)
        {
            savers[type] = (d, o) => SaveEmptyObject(d, type);
            loaders[type.Name] = n => CreateEmptyObject(type);
        }

        public static void AddStorage<T>(Func<XmlDocument, object, XmlElement> saver, Func<XmlNode, object> loader)
        {
            savers[typeof(T)] = saver;
            loaders[typeof(T).Name] = loader;
        }

        #region Models

        public static XmlElement SaveNetwork(XmlDocument doc, object o)
        {
            if (!(o is Network net))
                throw new Exception("object must be Network!");

            var el = doc.CreateElement("Network");

            return el;
        }

        public static Network LoadNetwork(XmlNode node)
        {
            var result = Activator.CreateInstance<Network>();


            return result;
        }

        #endregion


        #region Layers

        public static XmlElement SaveFullLayer(XmlDocument doc, object o)
        {
            if (!(o is FullLayer layer))
                throw new Exception("layer must be FullLayer!");

            var el = doc.CreateElement("FullLayer");

            AddChild(el, "UnitCount", layer.UnitCount);
            AddTensorChild(el, "Weights", layer.Weights);
            AddTensorChild(el, "Bias", layer.Bias);

            return el;
        }

        private static FullLayer LoadFullLayer(XmlNode node)
        {
            if (node == null) return null;

            if (!node.HasChildNodes)
                return null;

            var unitCount = GetIntValue(node, "UnitCount");
            var result = (FullLayer)Activator.CreateInstance(typeof(FullLayer), unitCount);

            var weights = GetTensorValue(node, "Weights");
            if (weights != null)
                result.SetWeights(weights);

            var bias = GetTensorValue(node, "Bias");
            if (bias != null)
                result.SetBias(bias);

            return result;
        }

        public static XmlElement SaveMaxPoolingLayer(XmlDocument doc, object o)
        {
            if (!(o is MaxPooling layer))
                throw new Exception("layer must be MaxPooling!");

            return SavePoolingLayer("MaxPooling", doc, layer);
        }

        public static XmlElement SaveMeanPoolingLayer(XmlDocument doc, object o)
        {
            if (!(o is MeanPooling layer))
                throw new Exception("layer must be MeanPooling!");

            return SavePoolingLayer("MeanPooling", doc, layer);
        }

        public static PoolingLayer LoadPoolingLayer(XmlNode node)
        {
            var rows = GetIntValue(node, "Rows");
            var columns = GetIntValue(node, "Columns");
            var rowStride = GetIntValue(node, "RowStride");
            var columnStride = GetIntValue(node, "ColumnStride");

            if (node.Name == "MaxPooling")
                return (MaxPooling)Activator.CreateInstance(typeof(MaxPooling), rows, columns, rowStride, columnStride);
            if (node.Name == "MeanPooling")
                return (MeanPooling)Activator.CreateInstance(typeof(MeanPooling), rows, columns, rowStride, columnStride);

            throw new Exception($"Unknown type {node.Name}!");
        }

        private static XmlElement SavePoolingLayer(string type, XmlDocument doc, PoolingLayer layer)
        {
            var el = doc.CreateElement(type);

            AddChild(el, "Rows", layer.Rows.ToString());
            AddChild(el, "Columns", layer.Columns.ToString());
            AddChild(el, "RowStride", layer.RowStride.ToString());
            AddChild(el, "ColumnStride", layer.ColumnStride.ToString());

            return el;
        }

        public static XmlElement SaveConvLayer(XmlDocument doc, object o)
        {
            if (!(o is ConvLayer layer))
                throw new Exception("layer must be ConvLayer!");

            var el = doc.CreateElement(o.GetType().Name);

            AddChild(el, "FilterCount", layer.FilterCount.ToString());
            AddChild(el, "FilterRows", layer.FilterRows.ToString());
            AddChild(el, "FilterColumns", layer.FilterColumns.ToString());
            AddChild(el, "RowStride", layer.RowStride.ToString());
            AddChild(el, "ColumnStride", layer.ColumnStride.ToString());
            AddChild(el, "RowPadding", layer.RowPadding.ToString());
            AddChild(el, "ColumnPadding", layer.ColumnPadding.ToString());
            AddChild(el, "PaddingValue", layer.PaddingValue.ToString());

            AddTensorChild(el, "Filters", layer.Filters);
            AddTensorChild(el, "Bias", layer.Bias);

            return el;
        }

        public static ConvLayer LoadConvLayer(XmlNode node)
        {
            var count = GetIntValue(node, "FilterCount");
            var rows = GetIntValue(node, "FilterRows");
            var columns = GetIntValue(node, "FilterColumns");
            var rowStride = GetIntValue(node, "RowStride");
            var columnStride = GetIntValue(node, "ColumnStride");
            var rowPadding = GetIntValue(node, "RowPadding");
            var columnPadding = GetIntValue(node, "ColumnPadding");
            var paddingValue = GetDoubleValue(node, "PaddingValue");

            var result = (ConvLayer)Activator.CreateInstance(typeof(ConvLayer), count, rows, columns, rowStride, columnStride, rowPadding, columnPadding);
            result.PaddingValue = paddingValue;

            var filters = GetTensorValue(node, "Filters");
            if (filters != null)
                result.SetFilters(filters);

            var bias = GetTensorValue(node, "Bias");
            if (bias != null)
                result.SetBias(bias);

            return result;
        }

        #endregion


        #region Maths

        public static XmlElement SaveTensor(XmlDocument doc, object o)
        {
            if (!(o is Tensor tensor))
                throw new Exception($"type error! need a Tensor but o is {o.GetType().Name}");

            var el = doc.CreateElement("Tensor");
            if (tensor == null)
                return el;

            AddArrayChild(el, "Shape", tensor.shape);
            AddArrayChild(el, "Values", tensor.GetRawValues());

            return el;
        }

        public static Tensor LoadTensor(XmlNode node)
        {
            if (node == null)
                return null;

            if (!node.HasChildNodes)
                return null;

            var shape = GetIntArray(node, "Shape");
            var values = GetDoubleArray(node, "Values");

            return (Tensor)Activator.CreateInstance(typeof(Tensor), values, shape);
        }

        #endregion

        #region Helper Methods

        internal static void AddChild(XmlElement father, string name, string value)
        {
            var el = father.OwnerDocument.CreateElement(name);
            el.InnerText = value;
            father.AppendChild(el);
        }

        internal static void AddChild(XmlElement father, string name, int value)
        {
            AddChild(father, name, value.ToString());
        }

        internal static void AddChild(XmlElement father, string name, double value)
        {
            AddChild(father, name, value.ToString());
        }

        internal static void AddTensorChild(XmlElement father, string name, Tensor tensor)
        {
            var el = father.OwnerDocument.CreateElement(name);
            var t = XmlTensorStorage.Save(father.OwnerDocument, tensor);
            el.AppendChild(t);
            father.AppendChild(el);
        }

        internal static void AddArrayChild<T>(XmlElement father, string name, T[] array)
        {
            var el = father.OwnerDocument.CreateElement(name);
            var text = string.Join(",", array);
            el.InnerText = text;
            father.AppendChild(el);
        }

        internal static int GetIntValue(XmlNode node, string path)
        {
            var n = node.SelectSingleNode(path);
            //if (n == null)
            //    return 0;

            return int.Parse(n.InnerText);
        }

        internal static double GetDoubleValue(XmlNode node, string path)
        {
            var n = node.SelectSingleNode(path);
            //if (n == null)
            //    return 0;

            return double.Parse(n.InnerText);
        }

        internal static string GetStringValue(XmlNode node, string path)
        {
            var n = node.SelectSingleNode(path);
            //if (n == null)
            //    return null;

            return n.InnerText;
        }

        internal static Tensor GetTensorValue(XmlNode node, string path)
        {
            var n = node.SelectSingleNode($"{path}/Tensor");
            return XmlTensorStorage.Load(n);
        }

        internal static double[] GetDoubleArray(XmlNode node, string path)
        {
            var n = node.SelectSingleNode(path);
            if (n == null)
                return null;

            var text = n.InnerText;
            if (string.IsNullOrEmpty(text))
                return null;

            var split = text.Split(',');
            var result = new double[split.Length];

            for (int i = 0; i < split.Length; i++)
            {
                result[i] = double.Parse(split[i]);
            }
            return result;
        }

        internal static int[] GetIntArray(XmlNode node, string path)
        {
            var n = node.SelectSingleNode(path);
            if (n == null)
                return null;

            var text = n.InnerText;
            if (string.IsNullOrEmpty(text))
                return null;

            var split = text.Split(',');
            var result = new int[split.Length];

            for (int i = 0; i < split.Length; i++)
            {
                result[i] = int.Parse(split[i]);
            }
            return result;
        }

        internal static XmlElement SaveEmptyObject<T>(XmlDocument doc)
        {
            var type = typeof(T);
            return SaveEmptyObject(doc, type);
        }

        internal static XmlElement SaveEmptyObject(XmlDocument doc, Type type)
        {
            var el = doc.CreateElement(type.Name);
            return el;
        }

        internal static T CreateEmptyObject<T>()
        {
            return Activator.CreateInstance<T>();
        }

        internal static object CreateEmptyObject(Type type)
        {
            return Activator.CreateInstance(type);
        }

        #endregion
    }
}
