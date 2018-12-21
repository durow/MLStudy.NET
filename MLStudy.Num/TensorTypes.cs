using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Num
{
    public static class TensorTypes
    {
        public const string Float = "Single";
        public const string Double = "Double";
        public const string Int = "Int32";
        public const string Long = "Int64";

        public static Tensor<T> Exchange<T>(Tensor<double> tensor)
            where T : struct
        {
            var type = typeof(T).Name;

            if (type == Double) return (Tensor<T>)(object)tensor;

            var fromSpan = tensor.RawValues;

            if (type == Float)
            {
                var result = Tensor.Empty<float>(tensor.Shape.shape);
                var toSpan = result.RawValues;

                for (int i = 0; i < fromSpan.Length; i++)
                {
                    toSpan[i] = (float)fromSpan[i];
                }
                return (Tensor<T>)(object)result;
            }

            if (type == Int)
            {
                var result = Tensor.Empty<int>(tensor.Shape.shape);
                var toSpan = result.RawValues;
                for (int i = 0; i < fromSpan.Length; i++)
                {
                    toSpan[i] = (int)fromSpan[i];
                }
                return (Tensor<T>)(object)result;
            }

            if (type == Long)
            {
                var result = Tensor.Empty<long>(tensor.Shape.shape);
                var toSpan = result.RawValues;
                for (int i = 0; i < fromSpan.Length; i++)
                {
                    toSpan[i] = (long)fromSpan[i];
                }
                return (Tensor<T>)(object)result;
            }

            throw new NotImplementedException($"not implemented type exchange from double to {type}!");
        }

        public static Tensor<T> Exchange<T>(Tensor<float> tensor)
            where T : struct
        {
            var type = typeof(T).Name;

            if (type == Float) return (Tensor<T>)(object)tensor;

            var fromSpan = tensor.RawValues;

            if (type == Double)
            {
                var result = Tensor.Empty<double>(tensor.Shape.shape);
                var toSpan = result.RawValues;

                for (int i = 0; i < fromSpan.Length; i++)
                {
                    toSpan[i] = fromSpan[i];
                }
                return (Tensor<T>)(object)result;
            }

            if (type == Int)
            {
                var result = Tensor.Empty<int>(tensor.Shape.shape);
                var toSpan = result.RawValues;
                for (int i = 0; i < fromSpan.Length; i++)
                {
                    toSpan[i] = (int)fromSpan[i];
                }
                return (Tensor<T>)(object)result;
            }

            if (type == Long)
            {
                var result = Tensor.Empty<long>(tensor.Shape.shape);
                var toSpan = result.RawValues;
                for (int i = 0; i < fromSpan.Length; i++)
                {
                    toSpan[i] = (long)fromSpan[i];
                }
                return (Tensor<T>)(object)result;
            }

            throw new NotImplementedException($"not implemented type exchange from double to {type}!");
        }

        public static Tensor<T> Exchange<T>(Tensor<int> tensor)
            where T : struct
        {
            var type = typeof(T).Name;

            if (type == Int) return (Tensor<T>)(object)tensor;

            var fromSpan = tensor.RawValues;

            if (type == Double)
            {
                var result = Tensor.Empty<double>(tensor.Shape.shape);
                var toSpan = result.RawValues;

                for (int i = 0; i < fromSpan.Length; i++)
                {
                    toSpan[i] = fromSpan[i];
                }
                return (Tensor<T>)(object)result;
            }

            if (type == Float)
            {
                var result = Tensor.Empty<float>(tensor.Shape.shape);
                var toSpan = result.RawValues;
                for (int i = 0; i < fromSpan.Length; i++)
                {
                    toSpan[i] = fromSpan[i];
                }
                return (Tensor<T>)(object)result;
            }

            if (type == Long)
            {
                var result = Tensor.Empty<long>(tensor.Shape.shape);
                var toSpan = result.RawValues;
                for (int i = 0; i < fromSpan.Length; i++)
                {
                    toSpan[i] = fromSpan[i];
                }
                return (Tensor<T>)(object)result;
            }

            throw new NotImplementedException($"not implemented type exchange from double to {type}!");
        }

        public static Tensor<T> Exchange<T>(Tensor<long> tensor)
            where T : struct
        {
            var type = typeof(T).Name;

            if (type == Long) return (Tensor<T>)(object)tensor;

            var fromSpan = tensor.RawValues;

            if (type == Double)
            {
                var result = Tensor.Empty<double>(tensor.Shape.shape);
                var toSpan = result.RawValues;

                for (int i = 0; i < fromSpan.Length; i++)
                {
                    toSpan[i] = fromSpan[i];
                }
                return (Tensor<T>)(object)result;
            }

            if (type == Float)
            {
                var result = Tensor.Empty<float>(tensor.Shape.shape);
                var toSpan = result.RawValues;
                for (int i = 0; i < fromSpan.Length; i++)
                {
                    toSpan[i] = fromSpan[i];
                }
                return (Tensor<T>)(object)result;
            }

            if (type == Int)
            {
                var result = Tensor.Empty<int>(tensor.Shape.shape);
                var toSpan = result.RawValues;
                for (int i = 0; i < fromSpan.Length; i++)
                {
                    toSpan[i] = (int)fromSpan[i];
                }
                return (Tensor<T>)(object)result;
            }

            throw new NotImplementedException($"not implemented type exchange from double to {type}!");
        }
    }
}
