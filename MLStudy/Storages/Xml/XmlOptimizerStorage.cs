using System;
using System.Collections.Generic;
using System.Text;
using System.Xml;

namespace MLStudy.Storages.Xml
{
    public class XmlGradientDescentStorage : XmlStorageBase<GradientDescent> 
    {
        public override XmlElement Save(XmlDocument doc, object o)
        {
            if (!(o is GradientDescent gd))
                throw new Exception("object must be GridentDescent");

            var el = GetRootElement(doc);
            XmlStorage.AddChild(el, "LearningRate", gd.LearningRate);

            return el;
        }

        public override object Load(XmlNode node)
        {
            var learningRate = XmlStorage.GetDoubleValue(node, "LearningRate");
            return Activator.CreateInstance(Type, learningRate);
        }
    }
}
