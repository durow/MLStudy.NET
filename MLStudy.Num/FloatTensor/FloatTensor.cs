using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Num
{
    public partial class FloatTensor : Tensor<float>
    {
        public FloatTensor(TensorData<float> data) : base(data)
        {
        }
    }
}
