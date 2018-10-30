using MLStudy.Regularizations;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public abstract class Regularization
    {
        #region Static

        public static Dictionary<string, Type> Dict { get; private set; }

        public Regularization()
        {
            Dict = new Dictionary<string, Type>();
            Registor<Lasso>("L1");
            Registor<Lasso>("Lasso");
            Registor<Ridge>("L2");
            Registor<Ridge>("Ridge");
        }

        public void Registor<T>(string name) where T: Regularization
        {
            Dict[name.ToUpper()] = typeof(T);
        }

        public Regularization Get(string name)
        {
            name = name.ToUpper();

            if (!Dict.ContainsKey(name))
                return null;

            return (Regularization)Activator.CreateInstance(Dict[name]);
        }

        #endregion

        public double Weight { get; set; } = 0.1;

        public abstract double GetValue(double weights);

        public abstract Vector GetValue(Vector weights);

        public abstract Matrix GetValue(Matrix weights);
    }

    public enum RegularTypes
    {
        L1,
        L2
    }
}
