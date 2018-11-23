using System;
using System.Collections.Generic;
using System.Data;
using System.Text;

namespace MLStudy
{
    public class Utilities
    {
        public static Random Rand = new Random();

        public static Tensor ProbabilityToCode(Tensor output)
        {
            var code = output.GetSameShape();
            ProbabilityToCode(output, code);
            return code;
        }

        public static void ProbabilityToCode(Tensor probability, Tensor code)
        {
            if (probability.shape[1] == 1)
                Tensor.Apply(probability, code, a => a > 0.5 ? 1 : 0);
            else
            {
                var buff = new double[probability.shape[1]];
                for (int i = 0; i < probability.shape[0]; i++)
                {
                    probability.GetByDim1(i, buff);
                    ComputeCode(buff);
                    Array.Copy(buff, 0, code.GetRawValues(), i * buff.Length, buff.Length);
                }
            }
        }

        public static void ComputeCode(double[] buff)
        {
            var max = buff[0];
            var maxIndex = 0;
            buff[0] = 1;
            for (int i = 1; i < buff.Length; i++)
            {
                if (buff[i] > max)
                {
                    max = buff[i];
                    buff[maxIndex] = 0;
                    maxIndex = i;
                    buff[maxIndex] = 1;
                }
                else
                {
                    buff[i] = 0;
                }
            }
        }

        public static int[] GetRandomDistinct(int min, int max, int count)
        {
            if (count > max - min)
                throw new Exception("count must <= max-min");

            if (count < 1)
                return new int[0];

            var range = max - min;
            var src = new int[max - min];
            int temp = 0;

            for (int i = 0; i < range - 1; i++)
            {
                var r = Rand.Next(min, max);
                var lastIndex = range - 1 - i;
                temp = src[r];
                src[r] = src[lastIndex];
                src[lastIndex] = temp;
            }

            for (int i = 0; i < count; i++)
            {
                var r = Rand.Next(min, max);
                src[r] = src[i];
            }
        }

        public static void Shuffle<T>(List<T> list)
        {
            int count = list.Count;
            T temp;

            for (int i = 0; i < count; i++)
            {
                int r = Rand.Next(0, count - i);
                int lastIndex = count - i - 1;
                temp = list[r];
                list[r] = list[lastIndex];
                list[lastIndex] = temp;
            }
        }

        public static void Shuffle<T>(T[] array)
        {
            int count = array.Length;
            T temp;

            for (int i = 0; i < count; i++)
            {
                int r = Rand.Next(0, count - i);
                int lastIndex = count - i - 1;
                temp = array[r];
                array[r] = array[lastIndex];
                array[lastIndex] = temp;
            }
        }


        public static void Shuffle(DataTable table)
        {
            var rows = table.Rows.Count;
            var temp = table.NewRow();

            for (int i = 0; i < rows; i++)
            {
                var r = Rand.Next(0, rows - 1);
                var lastIndex = rows - i - 1;

                CopyDataRow(table.Rows[r], temp);
                CopyDataRow(table.Rows[lastIndex], table.Rows[r]);
                CopyDataRow(temp, table.Rows[lastIndex]);
            }
        }

        private static void CopyDataRow(DataRow src, DataRow dest)
        {
            var columns = src.ItemArray.Length;

            for (int i = 0; i < columns; i++)
            {
                dest[i] = src[i];
            }
        }
        //public static List<int> GetRandomDistinct(int min, int max, int count)
        //{
        //    if (count < 1)
        //        return new List<int>();

        //    if (count > (max - min))
        //        throw new Exception("count must < max-min");

        //    if (count > (max - min) / 2)
        //    {
        //        var temp = GetRandomDistinct(min, max, max - min - count);
        //        var remain = new List<int>();
        //        for (int i = min; i < max; i++)
        //        {
        //            if (temp.Contains(i))
        //                continue;
        //            remain.Add(i);
        //        }
        //        return remain;
        //    }

        //    var rand = new Random(seed);
        //    var result = new List<int>();
        //    while (result.Count < count)
        //    {
        //        var r = rand.Next(min, max);
        //        if (!result.Contains(r))
        //            result.Add(r);
        //    }
        //    return result;
        //}
    }
}
