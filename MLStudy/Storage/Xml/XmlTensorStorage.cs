using System;
using System.Collections.Generic;
using System.Text;
using System.Xml;

namespace MLStudy.Storage
{
    public static class XmlTensorStorage
    {
        public static XmlElement Save(XmlDocument doc, Tensor tensor)
        {
            var el = doc.CreateElement("Tensor");
            if (tensor == null)
                return el;

            var shape = doc.CreateElement("Shape");
            var values = doc.CreateElement("Values");
            shape.InnerText = string.Join(",", tensor.shape);
            values.InnerText = string.Join(",", tensor.GetRawValues());
            el.AppendChild(shape);
            el.AppendChild(values);
            return el;
        }

        public static Tensor Load(XmlNode node)
        {
            if (!node.HasChildNodes)
                return null;

            var shapeText = node.SelectSingleNode("Shape").InnerText;
            var shape = GetShape(shapeText);
            var valuesText = node.SelectSingleNode("Values").InnerText;
            var values = GetRawValues(valuesText);

            return (Tensor)Activator.CreateInstance(typeof(Tensor), values, shape);
        }

        private static double[] GetRawValues(string text)
        {
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

        private static int[] GetShape(string text)
        {
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
    }
}
