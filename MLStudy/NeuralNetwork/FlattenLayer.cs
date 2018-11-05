using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class FlattenLayer
    {
        public int OriginRows { get; private set; }
        public int OriginColumns { get; private set; }
        public int OriginDepth { get; private set; }

        public Matrix FlattenToMatrix(Tensor3 t)
        {
            OriginRows = t.Rows;
            OriginColumns = t.Columns;
            OriginDepth = t.Depth;

            var result = new Matrix(1, OriginRows * OriginColumns * OriginDepth);
            var counter = 0;
            for (int i = 0; i < t.Depth; i++)
            {
                for (int j = 0; j < t.Rows; j++)
                {
                    for (int k = 0; k < t.Columns; k++)
                    {
                        result[1, counter] = t[i, j, k];
                        counter++;
                    }
                }
            }
            return result;
        }
    }
}
