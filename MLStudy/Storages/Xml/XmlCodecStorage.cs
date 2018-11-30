using MLStudy.PreProcessing;
using System;
using System.Collections.Generic;
using System.Text;
using System.Xml;

namespace MLStudy.Storages.Xml
{
    public class XmlOneHotCodecStorage : XmlStorageBase<OneHotCodec>
    {
        public override XmlElement Save(XmlDocument doc, object o)
        {
            if (o == null)
                return null;

            if (!(o is OneHotCodec codec))
                throw new Exception("Codec must be OneHotCodec!");

            var el = GetRootElement(doc);
            var text = string.Join(",", codec.Categories);
            el.InnerText = text;

            return el;
        }

        public override object Load(XmlNode node)
        {
            if (node == null) return null;
            var cate = CodecHelper.GetCategories(node);
            return Activator.CreateInstance(Type, cate);
        }
    }

    public class XmlDummyCodecStorage : XmlStorageBase<DummyCodec>
    {
        public override XmlElement Save(XmlDocument doc, object o)
        {
            if (o == null)
                return null;

            if (!(o is DummyCodec codec))
                throw new Exception("Codec must be DummyCodec!");

            var el = GetRootElement(doc);
            var text = string.Join(",", codec.Categories);
            el.InnerText = text;

            return el;
        }

        public override object Load(XmlNode node)
        {
            if (node == null) return null;
            var cate = CodecHelper.GetCategories(node);
            return Activator.CreateInstance(Type, cate);
        }
    }

    public class XmlMapCodecStorage : XmlStorageBase<MapCodec>
    {
        public override XmlElement Save(XmlDocument doc, object o)
        {
            if (o == null)
                return null;

            if (!(o is MapCodec codec))
                throw new Exception("Codec must be MapCodec!");

            var el = GetRootElement(doc);
            var text = string.Join(",", codec.Categories);
            el.InnerText = text;

            return el;
        }

        public override object Load(XmlNode node)
        {
            if (node == null) return null;
            var cate = CodecHelper.GetCategories(node);
            return Activator.CreateInstance(Type, cate);
        }
    }

    internal static class CodecHelper
    {
        internal static List<string> GetCategories(XmlNode node)
        {
            var split = node.InnerText.Split(',');
            var result = new List<string>();

            foreach (var item in split)
            {
                result.Add(item);
            }

            return result;
        }
    }
}
