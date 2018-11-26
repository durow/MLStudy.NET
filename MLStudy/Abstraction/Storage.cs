using MLStudy.Abstraction;
using MLStudy.Storages;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public abstract class Storage
    {
        private static Storage Instance;

        public abstract void SaveToFile(object o, string filename);
        public abstract T LoadFromFile<T>(string filename);

        static Storage()
        {
            UseStorage<XmlStorage>();
        }

        public static void UseStorage<T>() where T:Storage
        {
            UseStorage(Activator.CreateInstance<T>());
        }

        public static void UseStorage(Storage storage)
        {
            Instance = storage;
        }

        public static void Save(object o, string filename)
        {
            Instance.SaveToFile(o, filename);
        }

        public static T Load<T>(string filename)
        {
            return Instance.LoadFromFile<T>(filename);
        }
    }
}
