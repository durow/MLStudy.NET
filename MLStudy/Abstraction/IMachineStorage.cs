using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Abstraction
{
    public interface IMachineStorage
    {
        void Save(Machine machine, string filename);
        Machine Load(string filename);
    }
}
