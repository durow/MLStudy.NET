using System;
using System.Collections.Generic;
using System.Text;
using System.Xml;

namespace MLStudy.Storages
{
    public static class XmlTensorStorage
    {
        public static XmlElement Save(XmlDocument doc, Tensor tensor)
        {
            var el = doc.CreateElement("Tensor");
            if (tensor == null)
                return el;

            XmlStorage.AddArrayChild(el, "Shape", tensor.shape);
            XmlStorage.AddArrayChild(el, "Values", tensor.GetRawValues());

            return el;
        }

        public static Tensor Load(XmlNode node)
        {
            if (node == null)
                return null;

            if (!node.HasChildNodes)
                return null;

            var shape = XmlStorage.GetIntArray(node, "Shape");
            var values = XmlStorage.GetDoubleArray(node, "Values");

            return (Tensor)Activator.CreateInstance(typeof(Tensor), values, shape);
        }
    }
}
