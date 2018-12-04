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
        public TensorOld ForwardOutput { get; protected set; }
        public TensorOld BackwardOutput { get; protected set; }

        protected int samples;
        protected int channels;
        protected int outRows;
        protected int outColumns;

        public abstract TensorOld Backward(TensorOld error);

        public abstract ILayer CreateSame();

        public abstract TensorOld Forward(TensorOld input);

        public abstract TensorOld PreparePredict(TensorOld input);

        public abstract TensorOld PrepareTrain(TensorOld input);
    }
}
