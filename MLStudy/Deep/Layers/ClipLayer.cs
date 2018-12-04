using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Deep
{
    public class ClipLayer : ILayer
    {
        public string Name { get; set; }

        public double Min { get; set; }
        public double Max { get; set; }
        public ClipLayer(double min=0.000000001, double max = 0.999999999)
        {
            Min = min;
            Max = max;
        }

        public TensorOld Backward(TensorOld error)
        {
            return error;
        }

        public ILayer CreateSame()
        {
            return new ClipLayer(Min, Max);
        }

        public TensorOld Forward(TensorOld input)
        {
            input.Apply(a =>
            {
                if (a > Max)
                    return Max;
                if (a < Min)
                    return Min;
                return a;
            });
            return input;
        }

        public TensorOld PreparePredict(TensorOld input)
        {
            return input;
        }

        public TensorOld PrepareTrain(TensorOld input)
        {
            return input;
        }
    }
}
