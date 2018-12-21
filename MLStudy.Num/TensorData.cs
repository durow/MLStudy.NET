using System;

namespace MLStudy.Num
{
    public sealed class TensorData<T> where T : struct
    {
        private readonly T[] values;
        private Memory<T> mem;
        internal TensorShape shape;

        public Span<T> RawValues => mem.Span;
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

        public TensorData(T[] values)
            : this(values, values.Length)
        { }

        public TensorData(T[] values, params int[] shape)
        {
            this.shape = new TensorShape(shape);

            if (this.shape.TotalLength != values.Length)
                throw new Exception("values and shape are not match!");

            this.values = values;
            mem = values.AsMemory();
        }

        public TensorData(params int[] shape)
        {
            this.shape = new TensorShape(shape);
            values = new T[this.shape.TotalLength];
            mem = values.AsMemory();
        }

        public TensorData(T fillValue, int[] shape)
        {
            if (shape.Length == 0)
                throw new Exception("shape can't be empty!");

            this.shape = new TensorShape(shape);
            values = new T[this.shape.TotalLength];
            mem = values.AsMemory();
            mem.Span.Fill(fillValue);
        }

        private TensorData(T[] values, TensorShape shape, int start, int len)
        {
            this.values = values;
            this.shape = shape;
            mem = values.AsMemory(start, len);
        }

        public T GetValue(int[] index)
        {
            var offset = shape.GetOffset(index);
            return RawValues[offset];
        }

        public void SetValue(T value, int[] index)
        {
            var offset = shape.GetOffset(index);
            RawValues[offset] = value;
        }

        public TensorData<T> ReShape(int[] shape)
        {
            return new TensorData<T>(values, shape);
        }

        public TensorData<T> GetSubData(int[] index)
        {
            var offset = shape.GetOffset(index);
            var subShape = shape.GetSubShape(index);
            return new TensorData<T>(values, subShape, offset, subShape.TotalLength);
        }

        public Span<T> GetSubValues(int[] index)
        {
            var offset = shape.GetOffset(index);
            var len = shape.GetSubLength(index);
            return mem.Span.Slice(offset, len);
        }
    }
}
