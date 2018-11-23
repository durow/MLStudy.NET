using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class Utilities
    {
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
    }
}
