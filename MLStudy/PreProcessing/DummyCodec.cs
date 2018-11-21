using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.PreProcessing
{
    public class DummyCodec<T> : DiscreteCodec<T>
    {
        public int Length { get { return categories.Count - 1; } }

        public DummyCodec(List<T> categories)
            : base(categories)
        { }

        /// <summary>
        /// Dummy编码
        /// </summary>
        /// <param name="list">要编码的数据</param>
        /// <returns>编码结果</returns>
        public override Tensor Encode(List<T> list)
        {
            var result = new Tensor(list.Count, Length);

            for (int i = 0; i < list.Count; i++)
            {
                var index = categories.IndexOf(list[i]);
                if (index == -1)
                    throw new Exception($"{list[i]} is not in categories list!");

                var code = Index2Dummy(index);
                Array.Copy(code, 0, result.GetRawValues(), i * Length, Length);
            }

            return result;
        }

        /// <summary>
        /// Dummy解码
        /// </summary>
        /// <param name="t">要解码的数据</param>
        /// <returns>解码结果</returns>
        public override List<T> Decode(Tensor t)
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

        private int[] Index2Dummy(int index)
        {
            var result = new int[Length];
            for (int i = 0; i < Length; i++)
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
            return Length;
        }
    }
}
