/*
 * Description:这种归一化方法比较适用在数值比较集中的情况。这种方法两有个缺陷：
               当有新数据加入时，可能导致 max 和 min 发生变化，需要重新定义。
               如果max和min不稳定，很容易使得归一化结果不稳定，使得后续使用效果也不稳定。
 * Author:Yunxiao An
 * Date:2018.11.21
 */

using MLStudy.Abstraction;
using System;
using System.Linq;

namespace MLStudy.PreProcessing
{
    public class MinMaxNorm:INormalizer
    {
        public double Min { get; private set; }
        public double Max { get; private set; }
        public double Denom { get; private set; }

        public MinMaxNorm(Tensor tensor)
        {
            Min = tensor.Min();
            Max = tensor.Max();
            Denom = Max - Min;

            if (Denom == 0)
                throw new Exception("max=min");
        }

        public MinMaxNorm(double min, double max)
        {
            if (max >= min)
                throw new Exception("max<min");

            Max = max;
            Min = min;
            Denom = Max - Min;
        }

        public Tensor Normalize(Tensor input)
        {
            var result = input.GetSameShape();
            Normalize(input, result);
            return result;
        }

        public void Normalize(Tensor input, Tensor output)
        {
            Tensor.Apply(input, output, a => (a - Min) / Denom);
        }
    }
}
