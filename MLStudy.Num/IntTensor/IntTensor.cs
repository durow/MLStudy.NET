using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Num
{
    public class IntTensor : Tensor<int>
    {
        public IntTensor(TensorData<int> data) : base(data)
        {
        }

        public override Tensor<int> Add(int a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<int> Add(Tensor<int> a)
        {
            throw new NotImplementedException();
        }

        public override void AddLocal(int a)
        {
            throw new NotImplementedException();
        }

        public override void AddLocal(Tensor<int> a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<int> Divide(int a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<int> DivideBy(int a)
        {
            throw new NotImplementedException();
        }

        public override void DivideByLocal(int a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<int> DivideElementWise(Tensor<int> a)
        {
            throw new NotImplementedException();
        }

        public override void DivideElementWiseLocal(Tensor<int> a)
        {
            throw new NotImplementedException();
        }

        public override void DivideLocal(int a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<int> GetSubTensor(int[] index)
        {
            var subData = Values.GetSubData(index);
            return new IntTensor(subData);
        }

        public override Tensor<int> Minus(int a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<int> Minus(Tensor<int> a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<int> MinusBy(int a)
        {
            throw new NotImplementedException();
        }

        public override void MinusLocal(int a)
        {
            throw new NotImplementedException();
        }

        public override void MinusLocal(Tensor<int> a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<int> Multiple(int a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<int> Multiple(Tensor<int> a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<int> MultipleElementWise(Tensor<int> a)
        {
            throw new NotImplementedException();
        }

        public override void MultipleElementWiseLocal(Tensor<int> a)
        {
            throw new NotImplementedException();
        }

        public override void MultipleLocal(int a)
        {
            throw new NotImplementedException();
        }

        public override void MunusByLocal(int a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<int> ReShape(params int[] newShape)
        {
            var data = Values.ReShape(newShape);
            return new IntTensor(data);
        }
    }
}
