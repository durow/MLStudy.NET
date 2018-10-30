using MLStudy.Optimization;
using System;
using System.Collections.Generic;

namespace MLStudy
{
    public abstract class OutputLayer
    {
        //#region static

        //public static Dictionary<string, Type> Dict;

        //static OutputLayer()
        //{
        //    Dict = new Dictionary<string, Type>();
        //    Registor<LinearRegressionOut>("LinearRegression");
        //    Registor<LogisticRegressionOut>("LogisticRegression");
        //    Registor<SoftmaxOut>("Softmax");
        //}

        //public static void Registor<T>(string name) where T : OutputLayer
        //{
        //    Dict[name] = typeof(T);
        //}

        //public static OutputLayer Get(string name, int inputFeatures, int neuronCount)
        //{
        //    if (Dict.ContainsKey(name))
        //        return (OutputLayer)Activator.CreateInstance(Dict[name],);
        //    else
        //        return null;
        //}

        //#endregion

        public int InputFeatures { get; protected set; }
        public GradientOptimizer Optimizer { get; protected set; } = new NormalDescent();
        public WeightDecay WeightDecay { get; private set; }

        public Matrix ForwardInput { get; protected set; }
        public Matrix ForwardOutput { get; protected set; }
        public Matrix InputError { get; protected set; }
        public Matrix LinearError { get; protected set; }
        public double Loss { get; protected set; }

        public abstract Matrix Forward(Matrix input);

        public abstract Matrix Backward(Vector y);

        public abstract void AutoInitWeightsBias();

        public abstract double GetLoss(Matrix yHat, Vector y);

        public double GetLoss(Vector y)
        {
            return GetLoss(ForwardOutput, y);
        }

        public OutputLayer UseWeightDecay(WeightDecay decay)
        {
            WeightDecay = decay;
            return this;
        }

        public OutputLayer UseOptimizer(GradientOptimizer optimizer)
        {
            Optimizer = optimizer;
            return this;
        }

        public OutputLayer UseMomentum()
        {
            return this;
        }

        public OutputLayer UseAdam()
        {
            return this;
        }
    }

    public enum OutputTypes
    {
        Sigmoid,
        Regression,
        Softmax,
    }
}
