using System;
using System.Collections.Generic;
using System.Data;
using System.Text;

namespace MLStudy.Abstraction
{
    public interface IPreProcessor
    {
        Tensor PreProcessX(DataTable data);
        Tensor PreProcessY(DataTable data);
    }
}
