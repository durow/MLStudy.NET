using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Abstraction
{
    public interface IStorage
    {
        void SaveMachine(Machine machine, string filename);
        Machine LoadMachine(string filename);
        void SaveTrainer(Trainer trainer, string filename);
        Trainer LoadTrainer(string filename);
    }
}
