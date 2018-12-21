using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Num
{
    public partial class LongTensor : Tensor<long>
    {
        public LongTensor(TensorData<long> data) : base(data)
        {
        }
        
        public override Tensor<long> GetSubTensor(int[] index)
        {
            var subData = Values.GetSubData(index);
            return new LongTensor(subData);
        }

        public override Tensor<long> ReShape(params int[] newShape)
        {
            var data = Values.ReShape(newShape);
            return new LongTensor(data);
        }
    }
}
