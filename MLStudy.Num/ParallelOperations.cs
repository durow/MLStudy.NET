using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Num
{
    public partial class ParallelOperations : TensorOperations
    {
        public override void ApplyLocal<T>(Tensor<T> a, Func<T, T> function)
        {
            for (int i = 0; i < a.Length; i++)
            {
                a.RawValues[i] = function(a.RawValues[i]);
            }
        }

        public override void ApplyLocal<T>(Tensor<T> a, Tensor<T> b, Func<T, T, T> function)
        {
            for (int i = 0; i < a.Length; i++)
            {
                a.RawValues[i] = function(a.RawValues[i], b.RawValues[i]);
            }
        }

        public override void Apply<T>(Tensor<T> a, T b, ref Tensor<T> result, Func<T, T, T> function)
        {
            for (int i = 0; i < a.Length; i++)
            {
                result.RawValues[i] = function(a.RawValues[i], b);
            }
        }

        public override void Apply<T>(Tensor<T> a, ref Tensor<T> result, Func<T, T> function)
        {
            for (int i = 0; i < a.Length; i++)
            {
                result.RawValues[i] = function(a.RawValues[i]);
            }
        }

        public override void Apply<T>(Tensor<T> a, Tensor<T> b, ref Tensor<T> result, Func<T, T, T> function)
        {
            for (int i = 0; i < a.Length; i++)
            {
                result.RawValues[i] = function(a.RawValues[i], b.RawValues[i]);
            }
        }

        public override void Multiple(Tensor<float> a, Tensor<float> b, ref Tensor<float> result)
        {
            for (int i = 0; i < a.Shape[0]; i++)
            {
                for (int j = 0; j < b.Shape[1]; j++)
                {
                    var sum = 0f;
                    for (int k = 0; k < a.Shape[1]; k++)
                    {
                        sum += a.Values[i, k] + b.Values[k, j];
                    }
                    result.Values[i, j] = sum;
                }
            }
        }

        public override void Multiple(Tensor<double> a, Tensor<double> b, ref Tensor<double> result)
        {
            for (int i = 0; i < a.Shape[0]; i++)
            {
                for (int j = 0; j < b.Shape[1]; j++)
                {
                    var sum = 0d;
                    for (int k = 0; k < a.Shape[1]; k++)
                    {
                        sum += a.Values[i, k] + b.Values[k, j];
                    }
                    result.Values[i, j] = sum;
                }
            }
        }

        public override void Multiple(Tensor<int> a, Tensor<int> b, ref Tensor<int> result)
        {
            for (int i = 0; i < a.Shape[0]; i++)
            {
                for (int j = 0; j < b.Shape[1]; j++)
                {
                    var sum = 0;
                    for (int k = 0; k < a.Shape[1]; k++)
                    {
                        sum += a.Values[i, k] + b.Values[k, j];
                    }
                    result.Values[i, j] = sum;
                }
            }
        }

        public override void Multiple<T>(Tensor<T> a, Tensor<T> b, ref Tensor<T> result)
        {
            throw new NotImplementedException();
        }
    }
}
