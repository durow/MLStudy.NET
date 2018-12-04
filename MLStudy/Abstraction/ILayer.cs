using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Abstraction
{
    public interface ILayer
    {
        string Name { get; set; }

        TensorOld PrepareTrain(TensorOld input);
        TensorOld PreparePredict(TensorOld input);
        TensorOld Forward(TensorOld input);
        TensorOld Backward(TensorOld error);
        ILayer CreateSame();
    }
}
