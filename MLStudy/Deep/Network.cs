using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.Deep
{
    public sealed class Network
    {
        List<ILayer> Layers = new List<ILayer>();

        public Network AddFullLayer(int unitCount, int inputFeatures = -1)
        {
            return this;
        }

        public Network AddLayer(ILayer layer)
        {
            Layers.Add(layer);
            return this;
        }
    }
}
