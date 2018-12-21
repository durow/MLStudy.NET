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

        public override Tensor<float> Add(float a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<float> Add(Tensor<float> a)
        {
            throw new NotImplementedException();
        }

        public override void AddLocal(float a)
        {
            throw new NotImplementedException();
        }

        public override void AddLocal(Tensor<float> a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<float> Divide(float a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<float> DivideBy(float a)
        {
            throw new NotImplementedException();
        }

        public override void DivideByLocal(float a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<float> DivideElementWise(Tensor<float> a)
        {
            throw new NotImplementedException();
        }

        public override void DivideElementWiseLocal(Tensor<float> a)
        {
            throw new NotImplementedException();
        }

        public override void DivideLocal(float a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<float> Minus(float a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<float> Minus(Tensor<float> a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<float> MinusBy(float a)
        {
            throw new NotImplementedException();
        }

        public override void MinusLocal(float a)
        {
            throw new NotImplementedException();
        }

        public override void MinusLocal(Tensor<float> a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<float> Multiple(float a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<float> Multiple(Tensor<float> a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<float> MultipleElementWise(Tensor<float> a)
        {
            throw new NotImplementedException();
        }

        public override void MultipleElementWiseLocal(Tensor<float> a)
        {
            throw new NotImplementedException();
        }

        public override void MultipleLocal(float a)
        {
            throw new NotImplementedException();
        }

        public override void MunusByLocal(float a)
        {
            throw new NotImplementedException();
        }
    }
}
