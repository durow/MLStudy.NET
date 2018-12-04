using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class TensorGeneric
    {

        public static void Add(TensorGeneric<float> a, float b, TensorGeneric<float> result)
        {
            Apply(a, result, p => p + b);
        }

        public static TensorGeneric<float> Add(TensorGeneric<float> a, float b)
        {
            var result = a.GetSameShape();
            Add(a, b, result);
            return result;
        }

        public static TensorGeneric<T> Add<T>(TensorGeneric<T> a, T b)
        {
            var result = a.GetSameShape();
            Add(a, b, result);
            return result;
        }

        public static void Add<T>(TensorGeneric<T> a, T b, TensorGeneric<T> result)
        {
            throw new NotImplementedException($"not implement Add of {typeof(T).Name}");
        }

        public static void Apply<T>(TensorGeneric<T> a, TensorGeneric<T> result, Func<T, T> function)
        {
            var end = a.Values.startIndex + a.Values.Count;
            for (int i = a.Values.startIndex; i < end; i++)
            {
                result.Values.rawValues[i] = function(a.Values.rawValues[i]);
            }
        }

        public static void Apply<T>(TensorGeneric<T> a, TensorGeneric<T> b, TensorGeneric<T> result, Func<T, T, T> function)
        {
            var end = a.Values.startIndex + a.Values.Count;
            for (int i = a.Values.startIndex; i < end; i++)
            {
                result.Values.rawValues[i] = function(a.Values.rawValues[i], b.Values.rawValues[i]);
            }
        }
    }
}
