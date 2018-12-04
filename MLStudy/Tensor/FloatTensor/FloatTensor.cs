using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public sealed partial class Tensor : Tensor<float>
    {
        public Tensor(params int[] shape)
            : base(shape) { }

        public Tensor(TensorData<float> data)
            : base(data) { }

        public Tensor(float[] data)
        {
            Values = new TensorData<float>(data);
        }

        public override Tensor<float> ReShape(params int[] index)
        {
            var data = Values.ReShape(index);
            return new Tensor(data);
        }

        public override Tensor<float> GetSameShape()
        {
            return new Tensor(shape);
        }

        public override Tensor<float> GetTensor(params int[] index)
        {
            var data = Values.GetData(index);
            return new Tensor(data);
        }

        public override void AddLocal(float a)
        {
            TensorOperations.Instance.ApplyLocal(this, p => p + a);
        }

        public override Tensor<float> Add(float a)
        {
            var result = GetSameShape();
            TensorOperations.Instance.Apply(this, ref result, p => p + a);
            return result;
        }

        public override void AddLocal(Tensor<float> a)
        {
            TensorOperations.Instance.ApplyLocal(this, a, (m, n) => m + n);
        }

        public override Tensor<float> Add(Tensor<float> a)
        {
            var result = GetSameShape();
            TensorOperations.Instance.Apply(this, a, ref result, (m, n) => m + n);
            return result;
        }

        public override void MinusLocal(float a)
        {
            TensorOperations.Instance.ApplyLocal(this, p => p - a);
        }

        public override Tensor<float> Minus(float a)
        {
            var result = GetSameShape();
            TensorOperations.Instance.Apply(this, ref result, p => p - a);
            return result;
        }

        public override void MunusByLocal(float a)
        {
            TensorOperations.Instance.ApplyLocal(this, p => a - p);
        }

        public override Tensor<float> MinusBy(float a)
        {
            var result = GetSameShape();
            TensorOperations.Instance.Apply(this, ref result, p => a - p);
            return result;
        }

        public override void MinusLocal(Tensor<float> a)
        {
            TensorOperations.Instance.ApplyLocal(this, a, (m, n) => m - n);
        }

        public override Tensor<float> Minus(Tensor<float> a)
        {
            var result = GetSameShape();
            TensorOperations.Instance.Apply(this, a, ref result, (m, n) => m - n);
            return result;
        }

        public override void MultipleLocal(float a)
        {
            TensorOperations.Instance.ApplyLocal(this, p => a * p);
        }

        public override Tensor<float> Multiple(float a)
        {
            var result = GetSameShape();
            TensorOperations.Instance.Apply(this, ref result, p => a * p);
            return result;
        }

        public override Tensor<float> Multiple(Tensor<float> a)
        {
            var result = (Tensor<float>)Empty(shape[0], a.shape[1]);
            TensorOperations.Instance.Multiple(this, a, ref result);
            return result;
        }

        public override void MultipleElementWiseLocal(Tensor<float> a)
        {
            TensorOperations.Instance.ApplyLocal(this, a, (m, n) => m * n);
        }

        public override Tensor<float> MultipleElementWise(Tensor<float> a)
        {
            var result = a.GetSameShape();
            TensorOperations.Instance.Apply(this, a, ref result, (m, n) => m * n);
            return result;
        }

        public override void DivideLocal(float a)
        {
            TensorOperations.Instance.ApplyLocal(this, p => p / a);
        }

        public override Tensor<float> Divide(float a)
        {
            var result = GetSameShape();
            TensorOperations.Instance.Apply(this, ref result, p => p / a);
            return result;
        }

        public override void DivideByLocal(float a)
        {
            TensorOperations.Instance.ApplyLocal(this, p => a / p);
        }

        public override Tensor<float> DivideBy(float a)
        {
            var result = GetSameShape();
            TensorOperations.Instance.Apply(this, ref result, p => a / p);
            return result;
        }

        public override void DivideElementWiseLocal(Tensor<float> a)
        {
            TensorOperations.Instance.ApplyLocal(this, a, (m, n) => m / n);
        }

        public override Tensor<float> DivideElementWise(Tensor<float> a)
        {
            var result = a.GetSameShape();
            TensorOperations.Instance.Apply(this, a, ref result, (m, n) => m / n);
            return result;
        }
    }
}
