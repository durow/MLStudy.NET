using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Num
{
    public static partial class Tensor
    {
        public static Tensor<T> Empty<T>(params int[] shape)
            where T : struct
        {
            var type = typeof(T).Name;

            switch (type)
            {
                case SupportTypes.Float:
                    var fData = new TensorData<float>(shape);
                    var fTensor = new FloatTensor(fData);
                    return (Tensor<T>)(object)fTensor;
                case SupportTypes.Double:
                    var dData = new TensorData<double>(shape);
                    var dTensor = new DoubleTensor(dData);
                    return (Tensor<T>)(object)dTensor;
                case SupportTypes.Int:
                    var iData = new TensorData<int>(shape);
                    var iTensor = new IntTensor(iData);
                    return (Tensor<T>)(object)iTensor;
                default:
                    break;
            }
            throw new NotImplementedException($"type {type} not implemented!");
        }
    }
}
