using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public interface ILayer
    {
        Matrix Forward();
        Matrix Backward();
    }
}
