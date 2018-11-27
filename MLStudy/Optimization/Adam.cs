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
        private Dictionary<Tensor, AdamCache> dict = new Dictionary<Tensor, AdamCache>();

        public void Optimize(Tensor target, Tensor gradient)
        {
            if(!dict.ContainsKey(gradient))
            {
                dict[gradient] = new AdamCache(gradient.Shape);
            }

            var c = dict[gradient];

            Tensor.Apply(c.M, gradient, c.M, (m, g) => Beta1 * m + (1 - Beta1) * g);
            Tensor.Apply(c.V, gradient, c.V, (v, g) => Beta2 * v + (1 - Beta2) * g * g);
            Tensor.Apply(c.M, c.V, c.T, (m, v) => Alpha * m / (Math.Sqrt(v) + E));
            target.Minus(c.T);
        }
    }

    class AdamCache
    {
        public Tensor M { get; set; }
        public Tensor V { get; set; }
        public Tensor T { get; set; }

        public AdamCache(int[] shape)
        {
            M = new Tensor(shape);
            V = new Tensor(shape);
            T = new Tensor(shape);
        }
    }
}
