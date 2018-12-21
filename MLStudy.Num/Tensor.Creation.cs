using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Num
{
    public static partial class Tensor
    {
        public static Tensor<T> Values<T>(Array array)
            where T : struct
        {
            var shape = new TensorShape(TensorShape.GetShapeFromArray(array));

            var type = typeof(T).Name;

            switch (type)
            {
                case TensorTypes.Float:
                    var fData = new TensorData<float>(shape);
                    var fSpan = fData.RawValues;
                    for (int i = 0; i < fSpan.Length; i++)
                    {
                        fSpan[i] = (float)array.GetValue(shape.OffsetToIndex(i));
                    }
                    var fTensor = new FloatTensor(fData);
                    return (Tensor<T>)(object)fTensor;
                case TensorTypes.Double:
                    var dData = new TensorData<double>(shape);
                    var dSpan = dData.RawValues;
                    for (int i = 0; i < dSpan.Length; i++)
                    {
                        dSpan[i] = (double)array.GetValue(shape.OffsetToIndex(i));
                    }
                    var dTensor = new DoubleTensor(dData);
                    return (Tensor<T>)(object)dTensor;
                case TensorTypes.Int:
                    var iData = new TensorData<int>(shape);
                    var iSpan = iData.RawValues;
                    for (int i = 0; i < iSpan.Length; i++)
                    {
                        var index = shape.OffsetToIndex(i);
                        iSpan[i] = (int)array.GetValue(index);
                    }
                    var iTensor = new IntTensor(iData);
                    return (Tensor<T>)(object)iTensor;
                case TensorTypes.Long:
                    var lData = new TensorData<long>(shape);
                    var lSpan = lData.RawValues;
                    for (int i = 0; i < lSpan.Length; i++)
                    {
                        lSpan[i] = (long)array.GetValue(shape.OffsetToIndex(i));
                    }
                    var lTensor = new LongTensor(lData);
                    return (Tensor<T>)(object)lTensor;
                default:
                    break;
            }
            throw new NotImplementedException($"type {type} not implemented!");
        }

        public static Tensor<T> Empty<T>(params int[] shape)
            where T : struct
        {
            var type = typeof(T).Name;

            switch (type)
            {
                case TensorTypes.Float:
                    var fData = new TensorData<float>(shape);
                    var fTensor = new FloatTensor(fData);
                    return (Tensor<T>)(object)fTensor;
                case TensorTypes.Double:
                    var dData = new TensorData<double>(shape);
                    var dTensor = new DoubleTensor(dData);
                    return (Tensor<T>)(object)dTensor;
                case TensorTypes.Int:
                    var iData = new TensorData<int>(shape);
                    var iTensor = new IntTensor(iData);
                    return (Tensor<T>)(object)iTensor;
                case TensorTypes.Long:
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
                case TensorTypes.Float:
                    var fTensor = Fill(1f, shape);
                    return (Tensor<T>)(object)fTensor;
                case TensorTypes.Double:
                    var dTensor = Fill(1d, shape);
                    return (Tensor<T>)(object)dTensor;
                case TensorTypes.Int:
                    var iTensor = Fill(1, shape);
                    return (Tensor<T>)(object)iTensor;
                case TensorTypes.Long:
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
                case TensorTypes.Float:
                    var fTensor = Fill(0f, shape);
                    return (Tensor<T>)(object)fTensor;
                case TensorTypes.Double:
                    var dTensor = Fill(0d, shape);
                    return (Tensor<T>)(object)dTensor;
                case TensorTypes.Int:
                    var iTensor = Fill(0, shape);
                    return (Tensor<T>)(object)iTensor;
                case TensorTypes.Long:
                    var lTensor = Fill(0L, shape);
                    return (Tensor<T>)(object)lTensor;
                default:
                    break;
            }
            throw new NotImplementedException($"type {type} not implemented!");
        }
    }
}
