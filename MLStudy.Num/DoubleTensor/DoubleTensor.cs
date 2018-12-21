using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Num
{
    public class DoubleTensor : Tensor<double>
    {
        public DoubleTensor(TensorData<double> data) : base(data)
        {
        }

        public override Tensor<double> Add(double a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<double> Add(Tensor<double> a)
        {
            throw new NotImplementedException();
        }

        public override void AddLocal(double a)
        {
            throw new NotImplementedException();
        }

        public override void AddLocal(Tensor<double> a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<double> Divide(double a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<double> DivideBy(double a)
        {
            throw new NotImplementedException();
        }

        public override void DivideByLocal(double a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<double> DivideElementWise(Tensor<double> a)
        {
            throw new NotImplementedException();
        }

        public override void DivideElementWiseLocal(Tensor<double> a)
        {
            throw new NotImplementedException();
        }

        public override void DivideLocal(double a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<double> GetSubTensor(int[] index)
        {
            var subData = Values.GetSubData(index);
            return new DoubleTensor(subData);
        }

        public override Tensor<double> Minus(double a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<double> Minus(Tensor<double> a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<double> MinusBy(double a)
        {
            throw new NotImplementedException();
        }

        public override void MinusLocal(double a)
        {
            throw new NotImplementedException();
        }

        public override void MinusLocal(Tensor<double> a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<double> Multiple(double a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<double> Multiple(Tensor<double> a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<double> MultipleElementWise(Tensor<double> a)
        {
            throw new NotImplementedException();
        }

        public override void MultipleElementWiseLocal(Tensor<double> a)
        {
            throw new NotImplementedException();
        }

        public override void MultipleLocal(double a)
        {
            throw new NotImplementedException();
        }

        public override void MunusByLocal(double a)
        {
            throw new NotImplementedException();
        }

        public override Tensor<double> ReShape(params int[] newShape)
        {
            var data = Values.ReShape(newShape);
            return new DoubleTensor(data);
        }
    }
}
