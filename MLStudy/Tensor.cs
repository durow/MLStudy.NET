using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class Tensor
    {
        private float[] values;
        private int[] shape;
        private int[] dimensionSize;

        public int Rank { get; private set; }
        public int[] Shape
        {
            get
            {
                return getShape();
            }
        }
        public float this[params int[] index]
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
        public Tensor this[int index]
        {
            get
            {
                return getRankOne(index);
            }
        }
        public long ElementCount
        {
            get
            {
                return values.Length;
            }
        }

        public Tensor(Array data)
        {
            var shape = new int[data.Rank];
            for (int i = 0; i < shape.Length; i++)
            {
                shape[i] = data.GetLength(i);
            }
            InitTensor(shape);

            values = new float[data.LongLength];
            for (int i = 0; i < data.LongLength; i++)
            {
                values[i] = (float)data.GetValue(getIndex(i));
            }
        }

        public Tensor(int dim1, params int[] moreDimensions)
        {
            var shape = new int[moreDimensions.Length + 1];
            shape[0] = dim1;
            for (int i = 0; i < moreDimensions.Length; i++)
            {
                shape[i + 1] = moreDimensions[i];
            }
            InitTensor(shape);
            values = new float[getTotalLength(shape)];
        }

        public Tensor(float[] data, params int[] shape)
        {
            var len = getTotalLength(shape);
            if (len != data.Length)
                throw new TensorShapeException("data length and shape are not same!");

            InitTensor(shape);
            values = data;
        }

        public float GetValue(params int[] index)
        {
            if (index.Length != Rank)
                throw new TensorShapeException("index must be the same length as Rank!");

            for (int i = 0; i < Rank; i++)
            {
                if (index[i] >= shape[i])
                    throw new TensorShapeException($"index out of range! index is {index}, shape is {shape}");
            }

            var offset = getOffset(index);
            return values[offset];
        }

        public void SetValue(float value, params int[] index)
        {
            if (index.Length != Rank)
                throw new TensorShapeException("index must be the same length as Rank!");

            for (int i = 0; i < Rank; i++)
            {
                if (index[i] >= shape[i])
                    throw new TensorShapeException($"index out of range! index is {index}, shape is {shape}");
            }

            var offset = getOffset(index);
            values[offset] = value;
        }

        public override bool Equals(object o)
        {
            if (!(o is Tensor tensor))
                return false;

            if (Rank != tensor.Rank)
                return false;

            for (int i = 0; i < Rank; i++)
            {
                if (shape[i] != shape[i])
                    return false;
            }

            for (int i = 0; i < values.Length; i++)
            {
                if (values[i] != tensor.values[i])
                    return false;
            }

            return true;
        }

        public override string ToString()
        {
            if(Rank == 1)
            {
                var content = string.Join(", ", values);
                return $"[{content}]";
            }
            else
            {
                var result = new List<string>();
                for (int i = 0; i < shape[0]; i++)
                {
                    result.Add(this[i].ToString());
                }
                var content = string.Join(",\n", result);
                return $"[{content}]";
            }
        }

        private Tensor getRankOne(int index)
        {
            if (index >= shape[0])
                throw new TensorShapeException($"index out of range! index is {index} rank1 is {shape[0]}");

            var len = dimensionSize[0];
            var start = index * len;
            var data = new float[len];
            var newShape = new int[Rank - 1];
            for (int i = 0; i < newShape.Length; i++)
            {
                newShape[i] = shape[i + 1];
            }

            for (int i = 0; i < len; i++)
            {
                data[i] = values[start + i];
            }

            return new Tensor(data, newShape);
        }

        private void InitTensor(int[] shape)
        {
            this.shape = shape;
            Rank = shape.Length;
            setDimensionSize();
        }

        private int[] getShape()
        {
            var result = new int[shape.Length];
            shape.CopyTo(result, 0);
            return result;
        }

        private static int getTotalLength(int[] shape)
        {
            var result = 1;
            for (int i = 0; i < shape.Length; i++)
            {
                result *= shape[i];
            }
            return result;
        }

        private void setDimensionSize()
        {
            if (Rank == 1)
                return;

            dimensionSize = new int[Rank - 1];
            for (int i = 0; i < dimensionSize.Length; i++)
            {
                var temp = 1;
                for (int j = i + 1; j < shape.Length; j++)
                {
                    temp *= shape[j];
                }
                dimensionSize[i] = temp;
            }
        }

        private int getOffset(int[] index)
        {
            if (index.Length == 1)
                return index[0];

            var result = 0;
            for (int i = 0; i < dimensionSize.Length; i++)
            {
                result += dimensionSize[i] * index[i];
            }
            result += index[index.Length - 1];

            return result;
        }

        private int[] getIndex(int offset)
        {
            var rest = offset;
            var result = new int[Rank];
            for (int i = 0; i < dimensionSize.Length; i++)
            {
                result[i] = rest / dimensionSize[i];
                rest = rest % dimensionSize[i];
            }
            result[Rank - 1] = rest;

            return result;
        }

        private void CheckShape(int[] shape)
        {
            if (shape.Length == 0)
                throw new TensorShapeException("Tensor rank must > 0 !");
            foreach (var item in shape)
            {
                if (item <= 0)
                    throw new TensorShapeException("Tensor shape must > 0 !");
            }
        }
    }
}
