using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace MLStudy
{
    public class MatrixParallel : MatrixOperations
    {
        #region Add

        #endregion


        #region Minus

        #endregion


        #region Multiple

        public override Matrix Multiple(Matrix a, Matrix b)
        {
            if (a.Rows * a.Columns * b.Columns < OperationLimit * MaxThreads)
                return base.Multiple(a, b);

            if (a.Columns != b.Rows)
                throw new Exception($"a.Columns={a.Columns} and b.Rows={b.Rows} are not equal!");

            var result = new Matrix(a.Rows, b.Columns);
            var opt = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };
            Parallel.For(0, a.Rows, ParallelOptions, i =>
            {
                Parallel.For(0, b.Columns, ParallelOptions, j =>
                {
                    for (int k = 0; k < a.Columns; k++)
                    {
                        result[i, j] += a[i, k] * b[k, j];
                    }
                });
            });

            return result;
        }

        public Matrix MultipleThreadLimit(Matrix a, Matrix b, int tasks)
        {
            if (a.Columns != b.Rows)
                throw new Exception($"a.Columns={a.Columns} and b.Rows={b.Rows} are not equal!");

            var result = new Matrix(a.Rows, b.Columns);

            Parallel.For(0, a.Rows,
                new ParallelOptions { MaxDegreeOfParallelism = tasks },
                i =>
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
            var opt = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };
            Parallel.For(0, a.Rows, ParallelOptions, i =>
            {
                Parallel.For(0, b.Columns, ParallelOptions, j =>
                {
                    for (int k = 0; k < a.Columns; k++)
                    {
                        result[i, j] += a[i, k] * b[k, j];
                    }
                });
            });

            return result;
        }

        public Matrix MultipleForeach(Matrix a, Matrix b)
        {
            if (a.Columns != b.Rows)
                throw new Exception($"a.Columns={a.Columns} and b.Rows={b.Rows} are not equal!");

            var result = new Matrix(a.Rows, b.Columns);

            var range = a.Rows / MaxThreads + 1;
            Parallel.ForEach(Partitioner.Create(0,a.Rows, range),
                counter =>
                {
                    for (int i = counter.Item1; i < counter.Item2; i++)
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

            return result;
        }

        #endregion


        #region Divide


        #endregion

        private (bool, int) GetRange(int rows, int columns)
        {
            var isRow = rows >= columns;
            var max = Math.Max(rows, columns);
            var range = max / MaxThreads + 1;
            return (isRow, range);
        }
    }
}
