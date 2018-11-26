using MLStudy.PreProcessing;
using System;
using System.Collections.Generic;
using System.Text;
using System.Xml;

namespace MLStudy.Storages.Xml
{
    //public class XmlOneHotStorage<T> : XmlStorageBase<OneHotCodec<T>>
    //{
    //    public override XmlElement Save(XmlDocument doc, object o)
    //    {
    //        if (o == null)
    //            return null;

    //        if (!(o is OneHotCodec<T> codec))
    //            throw new Exception("layer must be FullLayer!");

    //        var el = GetRootElement(doc);
    //        var text = string.Join(",", codec.Categories);
    //        el.InnerText = text;

    //        return el;
    //    }

    //    public override object Load(XmlNode node)
    //    {
    //        if (node == null) return null;

    //        var split = node.InnerText.Split(',');
    //        var gType = typeof(T);
            
    //        if(gType == typeof(int))
    //        {
    //            var result = new List<int>();
    //            foreach (var item in split)
    //            {
    //                result.Add(int.Parse(item));
    //            }
    //            return Activator.CreateInstance(Type, result);
    //        }

    //        if(gType == typeof(double))
    //        {
    //            var result = new List<double>();
    //            foreach (var item in split)
    //            {
    //                result.Add(double.Parse(item));
    //            }
    //            return Activator.CreateInstance(Type, result);
    //        }

    //        if(gType == typeof(string))
    //        {
    //            var result = new List<string>();
    //            foreach (var item in split)
    //            {
    //                result.Add(item);
    //            }
    //            return Activator.CreateInstance(Type, result);
    //        }

    //        var list = new List<T>();
    //        foreach (var item in split)
    //        {
    //            list.Add(Activator.CreateInstance<T>());
    //        }
    //        return list;
    //    }
    //}
}
