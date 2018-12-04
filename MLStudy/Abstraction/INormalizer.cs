using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Abstraction
{
    public interface INormalizer
    {
        TensorOld Normalize(TensorOld input);
        void Normalize(TensorOld input, TensorOld output);
    }
}
