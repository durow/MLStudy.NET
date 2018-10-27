using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace MLStudy
{
    public class KNearestNeighbor
    {
        public int K { get; set; } = 1;
        public Matrix X { get; private set; }
        public Vector y { get; private set; }
        public int L { get; set; } = 2;

        public double Error(Vector yHat, Vector y)
        {
            return LossFunctions.ErrorPercent(yHat, y);
        }

        public double Predict(Vector x)
        {
            var list = GetAllDistance(x);
            var nearest = GetKNearestNeighbors(list);
            return nearest.OrderByDescending(n => n.Value).First().Key;
        }

        public void Train(Matrix X, Vector y)
        {
            this.X = X;
            this.y = y;
        }

        private List<KNNResult> GetAllDistance(Vector x)
        {
            var list = new List<KNNResult>(X.Rows);
            for (int i = 0; i < X.Rows; i++)
            {
                var dist = Distance(x, X[i]);
                list.Add(new KNNResult(y[i], dist));
            }
            return list.OrderByDescending(d => d.Distance).ToList();
        }

        private Dictionary<double, int> GetKNearestNeighbors(List<KNNResult> list)
        {
            var dict = new Dictionary<double, int>();
            for (int i = 0; i < K; i++)
            {
                var label = list[i].Label;
                if (dict.ContainsKey(list[i].Label))
                    dict[label]++;
                else
                    dict[label] = 1;
            }
            return dict;
        }

        private double Distance(Vector x, Vector train)
        {
            var gap = x - train;
            if (L == 1)
                return gap.ApplyFunction(a => Math.Abs(a)).Sum();
            else
            {
                var t = gap.ApplyFunction(a => Math.Pow(a, L)).Sum();
                return Math.Pow(t, 1 / L);
            }
        }
    }

    class KNNResult
    {
        public double Label { get; private set; }
        public double Distance { get; private set; }

        public KNNResult(double label, double distance)
        {
            Label = label;
            Distance = distance;
        }
    }
}
