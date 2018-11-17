using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Abstraction
{
    public interface IOptimizer
    {
        void Optimize(Tensor src, Tensor graident);
    }
}
