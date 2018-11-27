using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Deep
{
    public class FlattenLayer : ILayer
    {
        public string Name { get; set; }
        private int[] lastInputShape;

        public Tensor Backward(Tensor error)
        {
            return GetUnFlatten(error);
        }

        public ILayer CreateSame()
        {
            return new FlattenLayer();
        }

        public Tensor Forward(Tensor input)
        {
            lastInputShape = input.shape;
            return GetFlatten(input);
        }

        public Tensor PreparePredict(Tensor input)
        {
            return GetFlatten(input);
        }

        public Tensor PrepareTrain(Tensor input)
        {
            return GetFlatten(input);
        }

        private Tensor GetFlatten(Tensor input)
        {
            var d1 = input.shape[0];
            return input.Reshape(d1, input.ElementCount / d1);
        }

        private Tensor GetUnFlatten(Tensor error)
        {
            return error.Reshape(lastInputShape);
        }
    }
}
