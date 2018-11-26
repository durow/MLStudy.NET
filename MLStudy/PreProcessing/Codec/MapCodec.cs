using System;
using System.Collections.Generic;
using MLStudy.Abstraction;
using System.Text;

namespace MLStudy.PreProcessing
{
    public class MapCodec<T>:DiscreteCodec<T>
    {
        public int CategoriesCount { get { return Categories.Count; } }
        /// <summary>
        /// Map编码的起始值
        /// </summary>
        public int MapStart { get; set; } = 0;
        /// <summary>
        /// Map编码的步长
        /// </summary>
        public int MapStep { get; set; } = 1;

        public MapCodec(IEnumerable<T> categories)
            : base(categories)
        { }

        /// <summary>
        /// Map编码
        /// </summary>
        /// <param name="list">要编码的数据</param>
        /// <returns>编码结果</returns>
        public override Tensor Encode(List<T> list)
        {
            var result = new Tensor(list.Count, 1);

            for (int i = 0; i < list.Count; i++)
            {
                var index = Categories.IndexOf(list[i]);
                if (index == -1)
                    throw new Exception($"{list[i]} is not in categories list!");

                var code = Index2Map(index);
                result.GetRawValues()[i] = code;
            }

            return result;
        }

        /// <summary>
        /// Map解码
        /// </summary>
        /// <param name="t">要解码的数据</param>
        /// <returns>解码结果</returns>
        public override List<T> Decode(Tensor t)
        {
            var result = new List<T>(t.ElementCount);
            for (int i = 0; i < t.shape[0]; i++)
            {
                var index = Map2Index((int)t.GetRawValues()[i]);
                result.Add(Categories[index]);
            }

            return result;
        }

        private int Index2Map(int index)
        {
            return MapStart + index * MapStep;
        }

        private int Map2Index(int map)
        {
            return (map - MapStart) / MapStep;
        }
    }
}
