using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Optimization
{
    public class Adam : IOptimizer
    {
        public double Beta1 { get; set; } = 0.9;
        public double Beta2 { get; set; } = 0.999;
        public double Alpha { get; set; } = 0.001;

        private readonly double E = 0.00000001;
        private Dictionary<TensorOld, AdamCache> dict = new Dictionary<TensorOld, AdamCache>();

        public void Optimize(TensorOld target, TensorOld gradient)
        {
            if(!dict.ContainsKey(gradient))
            {
                dict[gradient] = new AdamCache(gradient.Shape);
            }

            var c = dict[gradient];

            TensorOld.Apply(c.M, gradient, c.M, (m, g) => Beta1 * m + (1 - Beta1) * g);
            TensorOld.Apply(c.V, gradient, c.V, (v, g) => Beta2 * v + (1 - Beta2) * g * g);
            TensorOld.Apply(c.M, c.V, c.T, (m, v) => Alpha * m / (Math.Sqrt(v) + E));
            target.Minus(c.T);
        }
    }

    class AdamCache
    {
        public TensorOld M { get; set; }
        public TensorOld V { get; set; }
        public TensorOld T { get; set; }

        public AdamCache(int[] shape)
        {
            M = new TensorOld(shape);
            V = new TensorOld(shape);
            T = new TensorOld(shape);
        }
    }
}
