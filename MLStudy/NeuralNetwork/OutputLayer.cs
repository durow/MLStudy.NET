using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public abstract class OutputLayer
    {
        public int InputFeatures { get; protected set; }
        public double LearningRate
        {
            get
            {
                return Optimizer.LearningRate;
            }
            set
            {
                Optimizer.LearningRate = value;
            }
        }

        public GradientOptimizer Optimizer { get; protected set; } = new GradientOptimizer();

        public Matrix ForwardInput { get; protected set; }
        public Matrix ForwardOutput { get; protected set; }
        public Matrix InputError { get; protected set; }
        public Matrix LinearError { get; protected set; }
        public double Loss { get; protected set; }

        public abstract Matrix Forward(Matrix input);

        public abstract Matrix Backward(Vector y);

        public abstract void AutoInitWeightsBias();
    }
}
