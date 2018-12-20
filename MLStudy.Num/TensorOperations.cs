using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Num
{
    public abstract class TensorOperations
    {
        private static TensorOperations instance;
        public static TensorOperations Instance
        {
            get
            {
                if (instance == null)
                    instance = new SequenceOperations();
                return instance;
            }
        }

        public static void UseSequence()
        {
            instance = new SequenceOperations();
        }

        public static void UseParallel()
        {
            instance = new ParallelOperations();
        }

        public abstract void ApplyLocal<T>(Tensor<T> a, Func<T, T> function)
            where T : struct;

        public abstract void ApplyLocal<T>(Tensor<T> a, Tensor<T> b, Func<T, T, T> function)
            where T : struct;

        public abstract void Apply<T>(Tensor<T> a, T b, ref Tensor<T> result, Func<T, T, T> function)
            where T : struct;

        public abstract void Apply<T>(Tensor<T> a, ref Tensor<T> result, Func<T, T> function)
            where T : struct;

        public abstract void Apply<T>(Tensor<T> a, Tensor<T> b, ref Tensor<T> result, Func<T, T, T> function)
            where T : struct;

        public abstract void Multiple(Tensor<float> a, Tensor<float> b, ref Tensor<float> result);

        public abstract void Multiple(Tensor<double> a, Tensor<double> b, ref Tensor<double> result);

        public abstract void Multiple(Tensor<int> a, Tensor<int> b, ref Tensor<int> result);

        public abstract void Multiple<T>(Tensor<T> a, Tensor<T> b, ref Tensor<T> result)
            where T : struct;
    }
}
