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
                case SupportTypes.Long:
                    var lData = new TensorData<long>(shape);
                    var lTensor = new LongTensor(lData);
                    return (Tensor<T>)(object)lTensor;
                default:
                    break;
            }
            throw new NotImplementedException($"type {type} not implemented!");
        }

        public static Tensor<T> Ones<T>(params int[] shape)
            where T : struct
        {
            var type = typeof(T).Name;

            switch (type)
            {
                case SupportTypes.Float:
                    var fTensor = Fill(1f, shape);
                    return (Tensor<T>)(object)fTensor;
                case SupportTypes.Double:
                    var dTensor = Fill(1d, shape);
                    return (Tensor<T>)(object)dTensor;
                case SupportTypes.Int:
                    var iTensor = Fill(1, shape);
                    return (Tensor<T>)(object)iTensor;
                case SupportTypes.Long:
                    var lTensor = Fill(1L, shape);
                    return (Tensor<T>)(object)lTensor;
                default:
                    break;
            }
            throw new NotImplementedException($"type {type} not implemented!");
        }

        public static Tensor<T> Zeros<T>(params int[] shape)
            where T : struct
        {
            var type = typeof(T).Name;

            switch (type)
            {
                case SupportTypes.Float:
                    var fTensor = Fill(0f, shape);
                    return (Tensor<T>)(object)fTensor;
                case SupportTypes.Double:
                    var dTensor = Fill(0d, shape);
                    return (Tensor<T>)(object)dTensor;
                case SupportTypes.Int:
                    var iTensor = Fill(0, shape);
                    return (Tensor<T>)(object)iTensor;
                case SupportTypes.Long:
                    var lTensor = Fill(0L, shape);
                    return (Tensor<T>)(object)lTensor;
                default:
                    break;
            }
            throw new NotImplementedException($"type {type} not implemented!");
        }
    }
}
