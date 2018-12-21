using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Num
{
    public partial class FloatTensor : Tensor<float>
    {
        public FloatTensor(TensorData<float> data) : base(data)
        {}

        public override Tensor<To> AsType<To>()
        {
            return TensorTypes.Exchange<To>(this);
        }

        public override Tensor<float> GetSubTensor(int[] index)
        {
            var subData = Values.GetSubData(index);
            return new FloatTensor(subData);
        }

        public override Tensor<float> ReShape(params int[] newShape)
        {
            var data = Values.ReShape(newShape);
            return new FloatTensor(data);
        }
    }
}
