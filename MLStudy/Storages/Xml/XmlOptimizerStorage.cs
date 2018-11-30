using MLStudy.Optimization;
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

    public class XmlAdamStorage : XmlStorageBase<Adam>
    {
        public override XmlElement Save(XmlDocument doc, object o)
        {
            if (!(o is Adam adam))
                throw new Exception("object must be GridentDescent");

            var el = GetRootElement(doc);
            XmlStorage.AddChild(el, "Alpha", adam.Alpha);
            XmlStorage.AddChild(el, "Beta1", adam.Beta1);
            XmlStorage.AddChild(el, "Beta2", adam.Beta2);

            return el;
        }

        public override object Load(XmlNode node)
        {
            var result = (Adam)Activator.CreateInstance(Type);
            result.Alpha = XmlStorage.GetDoubleValue(node, "Alpha");
            result.Beta1 = XmlStorage.GetDoubleValue(node, "Beta1");
            result.Beta2 = XmlStorage.GetDoubleValue(node, "Beta2");
            return result;
        }
    }
}
