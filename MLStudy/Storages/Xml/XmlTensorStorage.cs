using MLStudy.Storages.Xml;
using System;
using System.Collections.Generic;
using System.Text;
using System.Xml;

namespace MLStudy.Storages
{
    public class XmlTensorStorage : XmlStorageBase<TensorOld>
    {
        public override XmlElement Save(XmlDocument doc, object o)
        {
            var el = doc.CreateElement("Tensor");

            if (o == null)
                return el;

            if (!(o is TensorOld tensor))
                throw new Exception("layer must be Tensor!");

            XmlStorage.AddArrayChild(el, "Shape", tensor.shape);
            XmlStorage.AddArrayChild(el, "Values", tensor.GetRawValues());

            return el;
        }

        public override object Load(XmlNode node)
        {
            if (node == null)
                return null;

            if (!node.HasChildNodes)
                return null;

            var shape = XmlStorage.GetIntArray(node, "Shape");
            var values = XmlStorage.GetDoubleArray(node, "Values");

            return Activator.CreateInstance(typeof(TensorOld), values, shape);
        }
    }
}
