/*
 * Description:辅助函数，当前不太好分类的暂时放在这里
 *             后面有了合适的归属再重构过去
 * Author:YunXiao An
 * Date:2018.11.23
 */

using System;
using System.Collections.Generic;
using System.Data;

namespace MLStudy
{
    public class Utilities
    {
        public static Random Rand = new Random();

        /// <summary>
        /// 分类问题中产生的如果是概率，需要把概率转换为编码结果
        /// 后面再解码得到最终的分类
        /// </summary>
        /// <param name="probability">概率</param>
        /// <returns>编码</returns>
        public static TensorOld ProbabilityToCode(TensorOld probability)
        {
            var code = probability.GetSameShape();
            ProbabilityToCode(probability, code);
            return code;
        }

        /// <summary>
        /// 分类问题中产生的如果是概率，需要把概率转换为编码结果
        /// 后面再解码得到最终的分类
        /// </summary>
        /// <param name="probability">概率</param>
        /// <param name="code">编码</param>
        public static void ProbabilityToCode(TensorOld probability, TensorOld code)
        {
            if (probability.Rank == 1)
                probability = probability.Reshape(1, probability.ElementCount);

            if (probability.Rank != 2)
                throw new Exception("to do codec, Rank must be 2");

            if (probability.shape[1] == 1)
                TensorOld.Apply(probability, code, a => a > 0.5 ? 1 : 0);
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

        private static void ComputeCode(double[] buff)
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

        /// <summary>
        /// 获取count个不重复的随机数，范围从min（包含）到max（不包含）
        /// </summary>
        /// <param name="min">最小值（包含）</param>
        /// <param name="max">最大值（不包含）</param>
        /// <param name="count">生成随机数的个数</param>
        /// <returns>生成结果</returns>
        public static int[] GetRandomDistinct(int min, int max, int count)
        {
            if (count > max - min)
                throw new Exception("count must <= max-min");

            if (count < 1)
                return new int[0];

            var result = new int[count];
            GetRandomDistinct(min, max, result);
            return result;
        }

        /// <summary>
        /// 获取不重复的随机数，范围从min（包含）到max（不包含）
        /// 结果写入result参数，写满后就停止
        /// </summary>
        /// <param name="min">最小值（包含）</param>
        /// <param name="max">最大值（不包含）</param>
        /// <param name="result">结果，要求result.Length >= max-min</param>
        public static void GetRandomDistinct(int min, int max, int[] result)
        {
            var count = result.Length;

            if (count > max - min)
                throw new Exception("count must <= max-min");

            if (count < 1)
                return;

            var range = max - min;
            var src = new int[max - min];
            int temp = 0;

            //init src data
            for (int i = 0; i < range; i++)
            {
                src[i] = min + i;
            }

            for (int i = 0; i < range; i++)
            {
                var r = Rand.Next(0, range - 1);
                var lastIndex = range - 1 - i;
                temp = src[r];
                src[r] = src[lastIndex];
                src[lastIndex] = temp;

                if (i >= count)
                    break;
            }

            Array.Copy(src, range - count, result, 0, count);
        }

        /// <summary>
        /// 获取从min（包含）到max（不包含）的不重复的随机值
        /// </summary>
        /// <param name="min">最小值（包含）</param>
        /// <param name="max">最大值（不包含）</param>
        /// <returns>随机值的结果</returns>
        public static int[] GetRandomDistinct(int min, int max)
        {
            return GetRandomDistinct(min, max, max - min);
        }

        /// <summary>
        /// 获取从0（包含）到range（不包含）的不重复的随机值
        /// </summary>
        /// <param name="range">随机值范围</param>
        /// <returns>随机值的结果</returns>
        public static int[] GetRandomDistinct(int range)
        {
            return GetRandomDistinct(0, range, range);
        }

        /// <summary>
        /// 随机化list中元素的顺序，作用于list本身
        /// </summary>
        /// <typeparam name="T">list中元素的类型</typeparam>
        /// <param name="list">元素列表</param>
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

        /// <summary>
        /// 随机化array中元素的顺序，作用于array本身
        /// </summary>
        /// <typeparam name="T">array中元素的类型</typeparam>
        /// <param name="array">元素数组</param>
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

        /// <summary>
        /// 随机化DataTable中DataRow的顺序，作用于DataTable本身
        /// </summary>
        /// <param name="table">要随机化的DataTable</param>
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
    }
}
