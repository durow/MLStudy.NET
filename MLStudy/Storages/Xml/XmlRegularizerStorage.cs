using MLStudy.Regularizations;
using System;
using System.Collections.Generic;
using System.Text;
using System.Xml;

namespace MLStudy.Storages.Xml
{
    public class XmlRidgeStorage : XmlStorageBase<Ridge>
    {
        public override XmlElement Save(XmlDocument doc, object o)
        {
            if (!(o is Ridge ridge))
                throw new Exception("object must be Ridage");

            var el = GetRootElement(doc);
            XmlStorage.AddChild(el, "Strength", ridge.Strength);

            return el;
        }

        public override object Load(XmlNode node)
        {
            var strength = XmlStorage.GetDoubleValue(node, "Strength");
            return Activator.CreateInstance(Type, strength);
        }
    }

    public class XmlLassoStorage : XmlStorageBase<Lasso>
    {
        public override XmlElement Save(XmlDocument doc, object o)
        {
            if (!(o is Lasso ridge))
                throw new Exception("object must be Lasso");

            var el = GetRootElement(doc);
            XmlStorage.AddChild(el, "Strength", ridge.Strength);

            return el;
        }

        public override object Load(XmlNode node)
        {
            var strength = XmlStorage.GetDoubleValue(node, "Strength");
            return Activator.CreateInstance(Type, strength);
        }
    }
}
