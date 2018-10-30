using MLStudy.Regularizations;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public abstract class WeightDecay
    {
        #region Static

        public static Dictionary<string, Type> Dict { get; private set; }

        public WeightDecay()
        {
            Dict = new Dictionary<string, Type>();
            Registor<Lasso>("L1");
            Registor<Lasso>("Lasso");
            Registor<Ridge>("L2");
            Registor<Ridge>("Ridge");
        }

        public static void Registor<T>(string name) where T: WeightDecay
        {
            Dict[name.ToUpper()] = typeof(T);
        }

        public static WeightDecay Get(string name)
        {
            name = name.ToUpper();

            if (!Dict.ContainsKey(name))
                return null;

            return (WeightDecay)Activator.CreateInstance(Dict[name]);
        }

        #endregion

        public double Weight { get; set; } = 0.1;

        public abstract double GetValue(double weights);

        public abstract Vector GetValue(Vector weights);

        public abstract Matrix GetValue(Matrix weights);
    }

    public enum WeightDecayType
    {
        None,
        L1,
        L2
    }
}
