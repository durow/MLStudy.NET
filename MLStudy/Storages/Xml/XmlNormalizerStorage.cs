using MLStudy.PreProcessing;
using System;
using System.Collections.Generic;
using System.Text;
using System.Xml;

namespace MLStudy.Storages.Xml
{
    public class XmlNormalizerStorage : XmlStorageBase<ZScoreNorm>
    {
        public override XmlElement Save(XmlDocument doc, object o)
        {
            if (o == null)
                return null;

            if (!(o is ZScoreNorm norm))
                throw new Exception("Codec must be OneHotCodec!");

            var el = GetRootElement(doc);

            XmlStorage.AddChild(el, "Mean", norm.Mean);
            XmlStorage.AddChild(el, "Delta", norm.Delta);

            return el;
        }

        public override object Load(XmlNode node)
        {
            if (node == null) return null;

            var mean = XmlStorage.GetDoubleValue(node, "Mean");
            var delta = XmlStorage.GetDoubleValue(node, "Delta");
            return Activator.CreateInstance(Type, mean, delta);
        }
    }
}
