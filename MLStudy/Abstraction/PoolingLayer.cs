using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Abstraction
{
    public abstract class PoolingLayer : ILayer
    {
        public string Name { get; set; }
        public int Rows { get; protected set; }
        public int Columns { get; protected set; }
        public int RowStride { get; protected set; }
        public int ColumnStride { get; protected set; }
        public Tensor ForwardOutput { get; protected set; }
        public Tensor BackwardOutput { get; protected set; }

        protected int samples;
        protected int channels;
        protected int outRows;
        protected int outColumns;

        public abstract Tensor Backward(Tensor error);

        public abstract ILayer CreateSame();

        public abstract Tensor Forward(Tensor input);

        public abstract Tensor PreparePredict(Tensor input);

        public abstract Tensor PrepareTrain(Tensor input);
    }
}
