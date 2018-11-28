using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Xml;
using MLStudy.Abstraction;

namespace MLStudy.Storages
{
    public class XmlStorage : Storage
    {
        private static Dictionary<Type, IXmlStorage> typeDict = new Dictionary<Type, IXmlStorage>();
        private static Dictionary<string, IXmlStorage> nameDict = new Dictionary<string, IXmlStorage>();

        static XmlStorage()
        {
            var ifType = typeof(IXmlStorage);
            var ass = Assembly.GetExecutingAssembly();
            var types = ass.GetTypes().Where(t=>!t.IsGenericType && t.GetInterfaces().Contains(ifType));
            foreach (var type in types)
            {
                IXmlStorage instance;
                if (type.IsGenericType)
                {
                    var gType = type.MakeGenericType(type.GetGenericTypeDefinition());
                    instance = (IXmlStorage)Activator.CreateInstance(gType);
                }
                else
                    instance = (IXmlStorage)Activator.CreateInstance(type);

                typeDict[instance.Type] = instance;
                nameDict[instance.Type.Name] = instance;
            }
        }

        public override void SaveToFile(object o, string filename)
        {
            var type = o.GetType();
            if (!typeDict.ContainsKey(type))
                throw new Exception($"can't find {type}'s saver!");

            var doc = new XmlDocument();
            var dec = doc.CreateXmlDeclaration("1.0", "utf-8", null);
            doc.AppendChild(dec);

            var root = typeDict[type].Save(doc, o);
            doc.AppendChild(root);
            doc.Save(filename);
        }

        public static XmlElement SaveToEl(XmlDocument doc, object o)
        {
            if (o == null)
                return null;

            var type = o.GetType();
            if (!typeDict.ContainsKey(type))
                throw new Exception($"can't find {type}'s saver!");

            return typeDict[type].Save(doc, o);
        }

        public override T LoadFromFile<T>(string filename)
        {
            var doc = new XmlDocument();
            doc.Load(filename);
            var root = doc.ChildNodes[1];
            return LoadFromNode<T>(root);
        }

        public static T LoadFromNode<T>(XmlNode node)
        {
            if (node == null)
                return default(T);

            var type = typeof(T);
            if (typeDict.ContainsKey(type))
                return (T)typeDict[type].Load(node);

            if (nameDict.ContainsKey(node.Name))
                return (T)nameDict[node.Name].Load(node);

            throw new Exception($"can't find {type}'s loader!");

        }

        internal static void AddChild(XmlElement father, string name, string value)
        {
            var el = father.OwnerDocument.CreateElement(name);
            el.InnerText = value;
            father.AppendChild(el);
        }

        internal static void AddChild(XmlElement father, string name, int value)
        {
            AddChild(father, name, value.ToString());
        }

        internal static void AddChild(XmlElement father, string name, double value)
        {
            AddChild(father, name, value.ToString());
        }

        internal static void AddChild(XmlElement father, XmlElement child)
        {
            if (father != null && child != null)
                father.AppendChild(child);
        }

        internal static void AddObjectChild(XmlElement father, string name, object o)
        {
            var el = father.OwnerDocument.CreateElement(name);
            var t = SaveToEl(father.OwnerDocument, o);
            AddChild(el, t);
            AddChild(father, el);
        }

        internal static void AddObjectChild(XmlElement father, object o)
        {
            var el = SaveToEl(father.OwnerDocument, o);
            AddChild(father, el);
        }

        internal static void AddArrayChild<T>(XmlElement father, string name, T[] array)
        {
            var el = father.OwnerDocument.CreateElement(name);
            var text = string.Join(",", array);
            el.InnerText = text;
            father.AppendChild(el);
        }

        internal static int GetIntValue(XmlNode node, string path)
        {
            var n = node.SelectSingleNode(path);
            //if (n == null)
            //    return 0;

            return int.Parse(n.InnerText);
        }

        internal static double GetDoubleValue(XmlNode node, string path)
        {
            var n = node.SelectSingleNode(path);
            //if (n == null)
            //    return 0;

            return double.Parse(n.InnerText);
        }

        internal static string GetStringValue(XmlNode node, string path)
        {
            var n = node.SelectSingleNode(path);
            //if (n == null)
            //    return null;

            return n.InnerText;
        }

        internal static T GetObjectValue<T>(XmlNode node, string path)
        {
            if (node == null) return default(T);
            if (!node.HasChildNodes) return default(T);

            var n = node.SelectSingleNode(path);
            if (n == null) return default(T);
            if (!n.HasChildNodes) return default(T);

            return LoadFromNode<T>(n.FirstChild);
        }

        internal static double[] GetDoubleArray(XmlNode node, string path)
        {
            var n = node.SelectSingleNode(path);
            if (n == null)
                return null;

            var text = n.InnerText;
            if (string.IsNullOrEmpty(text))
                return null;

            var split = text.Split(',');
            var result = new double[split.Length];

            for (int i = 0; i < split.Length; i++)
            {
                result[i] = double.Parse(split[i]);
            }
            return result;
        }

        internal static int[] GetIntArray(XmlNode node, string path)
        {
            var n = node.SelectSingleNode(path);
            if (n == null)
                return null;

            var text = n.InnerText;
            if (string.IsNullOrEmpty(text))
                return null;

            var split = text.Split(',');
            var result = new int[split.Length];

            for (int i = 0; i < split.Length; i++)
            {
                result[i] = int.Parse(split[i]);
            }
            return result;
        }
    }
}
