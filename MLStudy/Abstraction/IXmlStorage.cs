using System;
using System.Collections.Generic;
using System.Text;
using System.Xml;

namespace MLStudy.Abstraction
{
    public interface IXmlStorage
    {
        Type Type { get; }
        XmlElement Save(XmlDocument doc, object o);
        object Load(XmlNode node);
    }
}
