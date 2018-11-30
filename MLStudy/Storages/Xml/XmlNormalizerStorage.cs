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
            return base.Save(doc, o);
        }

        public override object Load(XmlNode node)
        {
            return base.Load(node);
        }
    }
}
