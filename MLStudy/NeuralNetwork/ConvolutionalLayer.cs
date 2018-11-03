using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace MLStudy
{
    public class ConvolutionalLayer
    {
        public List<Filter> Filters { get; set; }
        public List<Matrix> ForwardInput { get; private set; }
        public List<Matrix> ForwardOutput { get; private set; }


        public List<Matrix> Forward(List<Matrix> input)
        {
            var result = new List<Matrix>(Filters.Count);
            ForwardInput = input;
            Parallel.For(0, Filters.Count, i =>
            {
                result[i] = ForwardFilter(Filters[i]);
            });
            return result;
        }

        private Matrix ForwardFilter(Filter filter)
        {
            throw new NotImplementedException();
        }
    }

    public class Filter
    {
        public List<Matrix> FilterCube { get; private set; }
        public int Rows { get; private set; }
        public int Columns { get; private set; }
        public int Depth { get; private set; }
        public double Bias { get; private set; }

        public Filter(int rows, int columns, int depth)
        {
            Rows = rows;
            Columns = columns;
            Depth = depth;
        }

        public void AutoInitWeightsBias()
        {
        }

        public void SetWeights(List<Matrix> filterCube)
        {
            FilterCube = filterCube;
            Rows = filterCube[0].Rows;
            Columns = filterCube[0].Columns;
            Depth = filterCube.Count;
        }

        public void SetBias(double bias)
        {
            Bias = bias;
        }
    }
}
