/*
 * Description:离散编码，用于多分类问题
 *             目前实现了OneHot、Dummy、顺序映射编码。
 * Author:YunXiao An
 * Date:2015.11.20
 */


using System;
using System.Collections.Generic;
using System.Linq;

namespace MLStudy.PreProcessing
{
    /// <summary>
    /// OneHot编码
    /// </summary>
    /// <typeparam name="T">类型</typeparam>
    public class DiscreteCodec<T>
    {
        private List<T> categories;
        /// <summary>
        /// OneHot编码长度
        /// </summary>
        public int OneHotLength { get { return categories.Count; } }
        /// <summary>
        /// Dummy编码长度
        /// </summary>
        public int DummyLength { get { return categories.Count - 1; } }
        /// <summary>
        /// Map编码的起始值
        /// </summary>
        public int MapStart { get; set; } = 0;
        /// <summary>
        /// Map编码的步长
        /// </summary>
        public int MapStep { get; set; } = 1;

        /// <summary>
        /// 使用分类列表创建一个OneHot编码器
        /// </summary>
        /// <param name="categories">分类列表</param>
        public DiscreteCodec(IEnumerable<T> categories)
        {
            this.categories = categories.Distinct().ToList();
        }

        /// <summary>
        /// OneHot编码
        /// </summary>
        /// <param name="list">要编码的数据</param>
        /// <returns>编码结果</returns>
        public Tensor OneHotEncode(List<T> list)
        {
            var result = new Tensor(list.Count, OneHotLength);

            for (int i = 0; i < list.Count; i++)
            {
                var index = categories.IndexOf(list[i]);
                if (index == -1)
                    throw new Exception($"{list[i]} is not in categories list!");

                var code = Index2OneHot(index);
                Array.Copy(code, 0, result.GetRawValues(), i * OneHotLength, OneHotLength);
            }

            return result;
        }

        /// <summary>
        /// Dummy编码
        /// </summary>
        /// <param name="list">要编码的数据</param>
        /// <returns>编码结果</returns>
        public Tensor DummyEncode(List<T> list)
        {
            var result = new Tensor(list.Count, DummyLength);

            for (int i = 0; i < list.Count; i++)
            {
                var index = categories.IndexOf(list[i]);
                if (index == -1)
                    throw new Exception($"{list[i]} is not in categories list!");

                var code = Index2Dummy(index);
                Array.Copy(code, 0, result.GetRawValues(), i * DummyLength, DummyLength);
            }

            return result;
        }

        /// <summary>
        /// Map编码
        /// </summary>
        /// <param name="list">要编码的数据</param>
        /// <returns>编码结果</returns>
        public Tensor MapEncode(List<T> list)
        {
            var result = new Tensor(list.Count,1);

            for (int i = 0; i < list.Count; i++)
            {
                var index = categories.IndexOf(list[i]);
                if (index == -1)
                    throw new Exception($"{list[i]} is not in categories list!");

                var code = Index2Map(index);
                result.GetRawValues()[i] = code;
            }

            return result;
        }

        /// <summary>
        /// OneHost解码
        /// </summary>
        /// <param name="t">要解码的数据</param>
        /// <returns>解码结果</returns>
        public List<T> OneHotDecode(Tensor t)
        {
            if (t.Rank != 2)
                throw new TensorShapeException("one hot decode tensor.Rank must be 2!");

            var result = new List<T>(t.shape[0]);
            var buff = new double[t.shape[1]];
            for (int i = 0; i < t.shape[0]; i++)
            {
                t.GetByDim1(i, buff);
                var index = OneHot2Index(buff);
                result.Add(categories[index]);
            }

            return result;
        }

        /// <summary>
        /// Dummy解码
        /// </summary>
        /// <param name="t">要解码的数据</param>
        /// <returns>解码结果</returns>
        public List<T> DummyDecode(Tensor t)
        {
            if (t.Rank != 2)
                throw new TensorShapeException("one hot decode tensor.Rank must be 2!");

            var result = new List<T>(t.shape[0]);
            var buff = new double[t.shape[1]];
            for (int i = 0; i < t.shape[0]; i++)
            {
                t.GetByDim1(i, buff);
                var index = Dummy2Index(buff);
                result.Add(categories[index]);
            }

            return result;
        }

        /// <summary>
        /// Map解码
        /// </summary>
        /// <param name="t">要解码的数据</param>
        /// <returns>解码结果</returns>
        public List<T> MapDecode(Tensor t)
        {
            var result = new List<T>(t.ElementCount);
            for (int i = 0; i < t.shape[0]; i++)
            {
                var index = Map2Index((int)t.GetRawValues()[i]);
                result.Add(categories[index]);
            }

            return result;
        }

        private int[] Index2OneHot(int index)
        {
            var result = new int[OneHotLength];
            for (int i = 0; i < OneHotLength; i++)
            {
                if (i == index)
                {
                    result[i] = 1;
                    break;
                }
            }
            return result;
        }

        private int OneHot2Index(double[] onehot)
        {
            for (int i = 0; i < onehot.Length; i++)
            {
                if (onehot[i] == 1)
                    return i;
            }
            throw new Exception($"unknown one hot code {onehot}!");
        }

        private int[] Index2Dummy(int index)
        {
            var result = new int[DummyLength];
            for (int i = 0; i < DummyLength; i++)
            {
                if (i == index)
                {
                    result[i] = 1;
                    break;
                }
            }
            return result;
        }

        private int Dummy2Index(double[] dummy)
        {
            for (int i = 0; i < dummy.Length; i++)
            {
                if (dummy[i] == 1)
                    return i;
            }
            return DummyLength;
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
