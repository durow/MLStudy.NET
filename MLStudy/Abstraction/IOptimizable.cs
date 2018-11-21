using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Abstraction
{
    public interface IOptimizable
    {
        void Optimize(IOptimizer optimizer);
        void Regularize(IRegularizer regularizer);
        ILayer CreateMirror();
    }
}
