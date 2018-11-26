using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Text;
using System.Xml;

namespace MLStudy.Storages.Xml
{
    public class XmlStorageBase<T> : IXmlStorage
    {
        public Type Type { get; private set; }
        public string Name
        {
            get
            {
                return Type.Name;
            }
        }
        public XmlStorageBase()
        {
            Type = typeof(T);
        }

        public virtual object Load(XmlNode node)
        {
            return Activator.CreateInstance<T>();
        }

        public virtual XmlElement Save(XmlDocument doc, object o)
        {
            return doc.CreateElement(typeof(T).Name);
        }

        protected XmlElement GetRootElement(XmlDocument doc)
        {
            return doc.CreateElement(Name);
        }
    }
}
