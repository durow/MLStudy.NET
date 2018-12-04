using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class TensorOperations
    {
        public static TensorOperations Instance = new TensorOperations();

        public void UseParallel()
        { }

        public virtual void ApplyLocal<T>(Tensor<T> a, Func<T, T> function)
            where T : struct
        {
            var aStart = a.Values.startIndex;
            for (int i = 0; i < a.Count; i++)
            {
                a.Values.rawValues[aStart + i] = function(a.Values.rawValues[aStart + i]);
            }
        }

        public virtual void ApplyLocal<T>(Tensor<T> a, Tensor<T> b, Func<T, T, T> function)
            where T : struct
        {
            var aStart = a.Values.startIndex;
            var bStart = b.Values.startIndex;
            for (int i = 0; i < a.Count; i++)
            {
                a.Values.rawValues[aStart + i] = function(a.Values.rawValues[aStart + i], b.Values.rawValues[bStart + i]);
            }
        }

        public virtual void Apply<T>(Tensor<T> a, T b, ref Tensor<T> result, Func<T, T, T> function)
            where T : struct
        {
            var aStart = a.Values.startIndex;
            var resultStart = result.Values.startIndex;
            for (int i = 0; i < a.Count; i++)
            {
                result.Values.rawValues[resultStart + i] = function(a.Values.rawValues[aStart + i], b);
            }
        }

        public virtual void Apply<T>(Tensor<T> a, ref Tensor<T> result, Func<T, T> function)
            where T : struct
        {
            var aStart = a.Values.startIndex;
            var resultStart = result.Values.startIndex;
            for (int i = 0; i < a.Count; i++)
            {
                result.Values.rawValues[resultStart + i] = function(a.Values.rawValues[aStart + i]);
            }
        }

        public virtual void Apply<T>(Tensor<T> a, Tensor<T> b, ref Tensor<T> result, Func<T, T, T> function)
            where T : struct
        {
            var aStart = a.Values.startIndex;
            var bStart = b.Values.startIndex;
            var resultStart = result.Values.startIndex;
            for (int i = 0; i < a.Count; i++)
            {
                result.Values.rawValues[resultStart + i] = function(a.Values.rawValues[aStart + i], b.Values.rawValues[bStart + i]);
            }
        }

        public virtual void Multiple(Tensor<float> a, Tensor<float> b, ref Tensor<float> result)
        {
            for (int i = 0; i < a.shape[0]; i++)
            {
                for (int j = 0; j < b.shape[1]; j++)
                {
                    var sum = 0f;
                    for (int k = 0; k < a.shape[1]; k++)
                    {
                        sum += a.Values[i, k] + b.Values[k, j];
                    }
                    result.Values[i, j] = sum;
                }
            }
        }

        public virtual void Multiple(Tensor<double> a, Tensor<double> b, ref Tensor<double> result)
        {
            for (int i = 0; i < a.shape[0]; i++)
            {
                for (int j = 0; j < b.shape[1]; j++)
                {
                    var sum = 0d;
                    for (int k = 0; k < a.shape[1]; k++)
                    {
                        sum += a.Values[i, k] + b.Values[k, j];
                    }
                    result.Values[i, j] = sum;
                }
            }
        }

        public virtual void Multiple(Tensor<int> a, Tensor<int> b, ref Tensor<int> result)
        {
            for (int i = 0; i < a.shape[0]; i++)
            {
                for (int j = 0; j < b.shape[1]; j++)
                {
                    var sum = 0;
                    for (int k = 0; k < a.shape[1]; k++)
                    {
                        sum += a.Values[i, k] + b.Values[k, j];
                    }
                    result.Values[i, j] = sum;
                }
            }
        }

        public void Multiple<T>(Tensor<T> a, Tensor<T> b, ref Tensor<T> result)
            where T : struct
        {
            throw new NotImplementedException();
        }
    }
}
