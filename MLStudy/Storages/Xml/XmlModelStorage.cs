using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Text;
using System.Xml;

namespace MLStudy.Storages
{
    public static class XmlModelStorage
    {
        private static Dictionary<Type, Func<XmlDocument, IModel, XmlElement>> ModelSaver = new Dictionary<Type, Func<XmlDocument, IModel, XmlElement>>();

        private static Dictionary<string, Func<XmlNode, IModel>> ModelLoader = new Dictionary<string, Func<XmlNode, IModel>>();

        static XmlModelStorage()
        {
        }

        public static void Add<T>(Func<XmlDocument, IModel, XmlElement> saver, Func<XmlNode, IModel> loader)
        {
            var type = typeof(T);
            ModelSaver[type] = saver;
            ModelLoader[type.Name] = loader;
        }

        public static XmlElement Save<T>(XmlDocument doc, T model) where T : IModel
        {
            var type = typeof(T);

            if (!ModelSaver.ContainsKey(type))
                throw new Exception($"can't find {type.Name}'s saver!");

            return ModelSaver[type](doc, model);
        }

        public static IModel Load(XmlNode node)
        {
            var type = node.Name;
            if(!ModelLoader.ContainsKey(type))
                throw new Exception($"can't find {type}'s loader!");

            return ModelLoader[type](node);
        }

        

        
    }
}
