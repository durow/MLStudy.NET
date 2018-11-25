using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Text;
using System.Xml;

namespace MLStudy.Storage
{
    public static class XmlLossStorage
    {
        public static Dictionary<Type, Func<XmlDocument, LossFunction, XmlElement>> LossSaver = new Dictionary<Type, Func<XmlDocument, LossFunction, XmlElement>>();

        public static Dictionary<string, Func<XmlNode, LossFunction>> LossLoader = new Dictionary<string, Func<XmlNode, LossFunction>>();

        static XmlLossStorage()
        {
            AddEmpty<CrossEntropy>();
            AddEmpty<MeanSquareError>();
        }

        public static XmlElement Save(XmlDocument doc, LossFunction loss)
        {
            var type = loss.GetType();

            if (!LossSaver.ContainsKey(type))
                throw new Exception($"can't find {type.Name}'s saver!");

            return LossSaver[type](doc, loss);
        }

        public static LossFunction Load(XmlNode node)
        {
            var name = node.Name;
            if (!LossLoader.ContainsKey(name))
                throw new Exception($"can't find {name}'s loader!");

            return LossLoader[name](node);
        }

        public static void AddEmpty<T>() where T : LossFunction
        {
            var type = typeof(T);
            LossSaver[type] = (d, o) => SaveEmptyObject<T>(d);
            LossLoader[type.Name] = e => CreateEmptyObject<T>();
        }

        internal static XmlElement SaveEmptyObject<T>(XmlDocument doc)
        {
            var type = typeof(T);
            var el = doc.CreateElement(type.Name);
            return el;
        }

        internal static T CreateEmptyObject<T>()
        {
            return Activator.CreateInstance<T>();
        }
    }
}
