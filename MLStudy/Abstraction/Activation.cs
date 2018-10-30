using System;
using System.Collections.Generic;
using System.Text;
using System.Reflection;
using System.Linq;
using MLStudy.Activations;

namespace MLStudy
{
    public abstract class Activation
    {
        public static Dictionary<string, Type> Dict { get; private set; }

        static Activation()
        {
            Dict = new Dictionary<string, Type>();
            Registor<ReLU>("ReLU");
            Registor<Tanh>("Tanh");
            Registor<Sigmoid>("Sigmoid");
        }

        public static void Registor<T>(string name) where T:Activation
        {
            Dict[name] = typeof(T);
        }

        public static Activation Get(string name)
        {
            if (Dict.ContainsKey(name))
                return (Activation)Activator.CreateInstance(Dict[name]);
            else
                return null;
        }

        public abstract Matrix Forward(Matrix input);
        public abstract Matrix Backward(Matrix forwardOutput, Matrix outputError);
    }

    public enum ActivationTypes
    {
        ReLU,
        Tanh,
        Sigmoid,
        None,
    }
}
