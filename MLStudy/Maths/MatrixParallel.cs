using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace MLStudy
{
    public class MatrixParallel
    {
        public Matrix Multiple(Matrix a, Matrix b)
        {
            if (a.Columns != b.Rows)
                throw new Exception($"a.Columns={a.Columns} and b.Rows={b.Rows} are not equal!");

            var result = new Matrix(a.Rows, b.Columns);

            Parallel.For(0, a.Rows, i =>
            {
                for (int j = 0; j < b.Columns; j++)
                {
                    for (int k = 0; k < a.Columns; k++)
                    {
                        result[i, j] += a[i, k] * b[k, j];
                    }
                }
            });

            return result;
        }

        public Matrix MultipleTurbo(Matrix a, Matrix b)
        {
            if (a.Columns != b.Rows)
                throw new Exception($"a.Columns={a.Columns} and b.Rows={b.Rows} are not equal!");

            var result = new Matrix(a.Rows, b.Columns);

            Parallel.For(0, a.Rows, i =>
            {
                Parallel.For(0, b.Columns, j =>
                {
                    for (int k = 0; k < a.Columns; k++)
                    {
                        result[i, j] += a[i, k] * b[k, j];
                    }
                });
            });

            return result;
        }

        public Matrix MultipleFinal(Matrix a, Matrix b, int threadCount)
        {
            if (a.Columns != b.Rows)
                throw new Exception($"a.Columns={a.Columns} and b.Rows={b.Rows} are not equal!");

            var result = new Matrix(a.Rows, b.Columns);

            var parts = a.Rows / threadCount + 1;

            for (int p = 0; p < parts; p++)
            {
                Parallel.For(0, threadCount, t =>
                {
                    var i = p * threadCount + t;

                    if (i < a.Rows)
                    {
                        for (int j = 0; j < b.Columns; j++)
                        {
                            for (int k = 0; k < a.Columns; k++)
                            {
                                result[i, j] += a[i, k] * b[k, j];
                            }
                        }
                    }
                });
            }

            Parallel.For(0, parts, i =>
            {
                for (int j = 0; j < b.Columns; j++)
                {
                    for (int k = 0; k < a.Columns; k++)
                    {
                        result[i, j] += a[i, k] * b[k, j];
                    }
                }
            });

            return result;
        }
    }
}
