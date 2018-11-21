using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Abstraction
{
    public interface INormalizer
    {
        Tensor Normalize(Tensor input);
        void Normalize(Tensor input, Tensor output);
    }
}
