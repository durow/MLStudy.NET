using System;
using System.Collections.Generic;
using System.Text;
using System.Data;
using MLStudy.Abstraction;
using System.Linq;

namespace MLStudy
{
    public class Machine<T>
    {
        public INormalizer Normalizer { get; set; }
        public IEngine Engine { get; set; }
        public IPreProcessor PreProcessor { get; set; }
        public DiscreteCodec<T> LabelCodec { get; set; }
        public MachineType MachineType { get; private set; } = MachineType.Classification;
        public Machine(IEngine engine)
        {
            Engine = engine;
        }

        public Tensor LastRawResult { get; private set; }
        public Tensor LastResultCodec { get; set; }

        public List<T>Predict(Tensor X)
        {
            X = Normalize(X);
            LastRawResult = Engine.Predict(X);
            if (LabelCodec == null)
            {
                throw new Exception("you need a LabelDecoder!");
            }

            if (MachineType == MachineType.Classification)
            {
                LastResultCodec = OutputToCode(LastRawResult);
            }

            return LabelCodec.Decode(LastResultCodec);
        }

        public List<T> Predict(DataTable table)
        {
            var x = PreProcessor.PreProcess(table);
            return Predict(x);
        }

        protected Tensor Normalize(Tensor input)
        {
            if (Normalizer == null)
                return input;

            return Normalizer.Normalize(input);
        }

        protected Tensor OutputToCode(Tensor output)
        {
            var code = output.GetSameShape();
            if (output.shape[1] == 1)
                Tensor.Apply(output, code, a => a > 0.5 ? 1 : 0);
            else
            {
                var buff = new double[output.shape[1]];
                for (int i = 0; i < output.shape[0]; i++)
                {
                    output.GetByDim1(i, buff);
                    ComputeCode(buff);
                    Array.Copy(buff, 0, code.GetRawValues(), i * buff.Length, buff.Length);
                }
            }
            return code;
        }

        private void ComputeCode(double[] buff)
        {
            var max = buff[0];
            var maxIndex = 0;
            buff[0] = 1;
            for (int i = 1; i < buff.Length; i++)
            {
                if(buff[i] > max)
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

    public enum MachineType
    {
        Regression,
        Classification
    }
}
