using System;

namespace MLStudy.Num
{
    public struct TensorShape
    {
        internal readonly int[] dimSize;
        internal readonly int[] shape;

        public readonly int Rank;
        public readonly int TotalLength;
        public int this[int index]
        {
            get
            {
                return shape[index];
            }
        }

        public ReadOnlySpan<int> Shape => new ReadOnlySpan<int>(shape);
        public ReadOnlySpan<int> DimSize => new ReadOnlySpan<int>(dimSize);

        public TensorShape(int[] shape)
        {
            if (shape.Length == 0)
                throw new Exception("shape can't empty!");

            this.shape = shape;
            dimSize = ComputeDimSize(shape);
            Rank = shape.Length;
            TotalLength = ComputeTotalLength(shape);
        }

        public int GetOffset(int[] index)
        {
            var result = 0;
            for (int i = 0; i < index.Length; i++)
            {
                result += dimSize[i] * index[i];
            }
            return result;
        }

        internal TensorShape GetSubShape(int[] index)
        {
            var indexLength = index.Length;

            if (indexLength == shape.Length)
                return new TensorShape(new int[] { 1 });

            var newRank = shape.Length - indexLength;
            var result = new int[newRank];
            Array.Copy(shape, indexLength, result, 0, newRank);
            return new TensorShape(result);
        }

        internal int GetSubLength(int[] index)
        {
            if (index.Length == Rank)
                return 0;

            return dimSize[index.Length - 1];
        }

        private static int[] ComputeDimSize(int[] shape)
        {
            var result = new int[shape.Length];
            for (int i = 0; i < result.Length; i++)
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

        private static int ComputeTotalLength(int[] shape)
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
    }
}
