using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class TensorData<T> where T : struct
    {
        internal T[] rawValues;
        internal int[] shape;
        internal int[] dimensionSize;
        internal int startIndex = 0;
        public int Count { get; private set; }

        public T this[params int[] index]
        {
            get
            {
                return GetValue(index);
            }
            set
            {
                SetValue(value, index);
            }
        }

        private TensorData() { }

        public TensorData(params int[] shape)
        {
            InitShapeInfo(shape);
            Count = GetTotalLength(shape);
            rawValues = new T[Count];
        }

        public TensorData(T[] data)
        {
            var shape = new int[] { 1, data.Length };
            InitShapeInfo(shape);
            Count = GetTotalLength(shape);
            rawValues = data;
        }

        public T GetValue(params int[] index)
        {
            var offset = GetOffset(index);
            return rawValues[offset];
        }

        public void SetValue(T value, params int[] index)
        {
            var offset = GetOffset(index);
            rawValues[offset] = value;
        }

        public TensorData<T> GetData(params int[] index)
        {
            var offset = GetOffset(index);
            var len = dimensionSize[index.Length - 1];
            var newShape = GetNewShape(index.Length);
            var newDimSize = GetNewDimensionSize(index.Length);

            var result = new TensorData<T>
            {
                rawValues = rawValues,
                startIndex = offset,
                Count = len,
                shape = newShape,
                dimensionSize = newDimSize,
            };

            return result;
        }

        public void SetData(T value, params int[] index)
        {
            var offset = GetOffset(index);
            var len = GetDimensionSize(index);

            for (int i = 0; i < len; i++)
            {
                rawValues[offset + i] = value;
            }
        }

        public void SetAll(T value)
        {
            for (int i = 0; i < rawValues.Length; i++)
            {
                rawValues[i] = value;
            }
        }

        internal TensorData<T> ReShape(params int[] shape)
        {
            var result = new TensorData<T>
            {
                rawValues = rawValues,
                startIndex = startIndex,
                shape = shape,
                Count = Count,
                dimensionSize = ComputeDimSize(shape)
            };
            return result;
        }

        internal int GetOffset(params int[] index)
        {
            var result = 0;
            for (int i = 0; i < index.Length; i++)
            {
                result += dimensionSize[i] * index[i];
            }
            return result + startIndex;
        }

        internal int[] GetNewShape(int indexLength)
        {
            if (indexLength == shape.Length)
                return new int[] { 1 };

            var newRank = shape.Length - indexLength;
            var result = new int[newRank];
            Array.Copy(shape, indexLength, result, 0, newRank);
            return result;
        }

        internal int[] GetNewDimensionSize(int indexLength)
        {
            if (indexLength == shape.Length)
                return new int[] { 1 };

            var newRank = shape.Length - indexLength;
            var result = new int[newRank];
            Array.Copy(dimensionSize, indexLength, result, 0, newRank);
            return result;
        }

        private void SetDimensionSize()
        {
            dimensionSize = ComputeDimSize(shape);
        }

        private int[] ComputeDimSize(int[] shape)
        {
            var result = new int[shape.Length];
            for (int i = 0; i < dimensionSize.Length; i++)
            {
                var temp = 1;
                for (int j = i + 1; j < shape.Length; j++)
                {
                    temp *= shape[j];
                }
                result[i] = temp;
            }
            return result;
        }

        private int GetDimensionSize(params int[] index)
        {
            if (index.Length == 0)
                return Count;

            return dimensionSize[index.Length - 1];
        }

        private static int GetTotalLength(int[] shape)
        {
            if (shape.Length == 0)
                return 1;

            var result = 1;
            for (int i = 0; i < shape.Length; i++)
            {
                result *= shape[i];
            }
            return result;
        }

        private void InitShapeInfo(int[] shape)
        {
            CheckShape(shape);
            this.shape = shape;
            SetDimensionSize();
        }

        private static void CheckShape(int[] shape)
        {
            foreach (var item in shape)
            {
                if (item <= 0)
                    throw new TensorShapeException("Tensor shape must > 0 !");
            }
        }
    }
}
