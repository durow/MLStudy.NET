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

        public TensorOld Backward(TensorOld error)
        {
            return GetUnFlatten(error);
        }

        public ILayer CreateSame()
        {
            return new FlattenLayer();
        }

        public TensorOld Forward(TensorOld input)
        {
            lastInputShape = input.shape;
            return GetFlatten(input);
        }

        public TensorOld PreparePredict(TensorOld input)
        {
            return GetFlatten(input);
        }

        public TensorOld PrepareTrain(TensorOld input)
        {
            return GetFlatten(input);
        }

        private TensorOld GetFlatten(TensorOld input)
        {
            var d1 = input.shape[0];
            return input.Reshape(d1, input.ElementCount / d1);
        }

        private TensorOld GetUnFlatten(TensorOld error)
        {
            return error.Reshape(lastInputShape);
        }
    }
}
