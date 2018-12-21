using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Num
{
    public partial class DoubleTensor : Tensor<double>
    {
        public DoubleTensor(TensorData<double> data) : base(data)
        {
        }
        
        public override Tensor<double> GetSubTensor(int[] index)
        {
            var subData = Values.GetSubData(index);
            return new DoubleTensor(subData);
        }

        public override Tensor<double> ReShape(params int[] newShape)
        {
            var data = Values.ReShape(newShape);
            return new DoubleTensor(data);
        }
    }
}
