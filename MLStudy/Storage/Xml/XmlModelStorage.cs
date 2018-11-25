using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Text;
using System.Xml;

namespace MLStudy.Storage
{
    public static class XmlModelStorage
    {
        public static Dictionary<Type, Func<IModel, XmlElement>> ModelSaver = new Dictionary<Type, Func<IModel, XmlElement>>();

        public static Dictionary<string, Func<XmlElement, IModel>> ModelLoader = new Dictionary<string, Func<XmlElement, IModel>>();





        static XmlModelStorage()
        {
        }

        public static XmlElement Save<T>(T model) where T : IModel
        {
            var type = typeof(T);

            if (!ModelSaver.ContainsKey(type))
                throw new Exception($"can't find {type.Name}'s saver!");

            return ModelSaver[type](model);
        }

        

        

        
    }
}
