using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Num
{
    public partial class IntTensor : Tensor<int>
    {
        public IntTensor(TensorData<int> data) : base(data)
        {}

        public override Tensor<To> AsType<To>()
        {
            return TensorTypes.Exchange<To>(this);
        }

        public override Tensor<int> GetSubTensor(int[] index)
        {
            var subData = Values.GetSubData(index);
            return new IntTensor(subData);
        }
        
        public override Tensor<int> ReShape(params int[] newShape)
        {
            var data = Values.ReShape(newShape);
            return new IntTensor(data);
        }
    }
}
