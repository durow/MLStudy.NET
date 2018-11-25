using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Abstraction
{
    public interface ITrainerStorage
    {
        void Save(Trainer trainer, string filename);
        Trainer Load(string filename);
    }
}
