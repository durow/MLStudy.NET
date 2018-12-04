using System;
using System.Collections.Generic;
using System.Data;
using System.Text;

namespace MLStudy.Abstraction
{
    public interface IPreProcessor
    {
        TensorOld PreProcessX(DataTable data);
        TensorOld PreProcessY(DataTable data);
    }
}
