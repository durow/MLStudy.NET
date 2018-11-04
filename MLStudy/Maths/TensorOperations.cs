using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLStudy
{
    public class TensorOperations
    {
        public static int OperationLimit { get; set; } = 10000;
        public static int MaxThreads
        {
            get
            {
                return ParallelOptions.MaxDegreeOfParallelism;
            }
            set
            {
                ParallelOptions.MaxDegreeOfParallelism = value;
            }
        }
        public static ParallelOptions ParallelOptions { get; private set; } = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };
        public static TensorOperations Instance { get; private set; } = new TensorOperations();

        public static void UseSequence()
        {
            Instance = new TensorOperations();
        }

        public static void UseParallel()
        { }

        #region Add

        public Vector Add(Vector v, double b)
        {
            var result = new double[v.Length];
            for (int i = 0; i < v.Length; i++)
            {
                result[i] = v[i] + b;
            }
            return new Vector(result);
        }

        public Vector Add(Vector a, Vector b)
        {
            CheckShape(a, b);

            var result = new double[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                result[i] = a[i] + b[i];
            }
            return new Vector(result);
        }

        public virtual Matrix Add(Matrix m, double b)
        {
            var result = new double[m.Rows, m.Columns];
            for (int i = 0; i < m.Rows; i++)
            {
                for (int j = 0; j < m.Columns; j++)
                {
                    result[i, j] = m[i, j] + b;
                }
            }
            return new Matrix(result);
        }

        public virtual Matrix Add(Matrix a, Vector b)
        {
            if (a.Rows != b.Length)
                throw new Exception("not the same size!");

            var result = new Matrix(a.Rows, a.Columns);
            for (int i = 0; i < a.Columns; i++)
            {
                for (int j = 0; j < a.Rows; j++)
                {
                    result[j, i] = a[j, i] + b[j];
                }
            }

            return result;
        }

        public virtual Matrix Add(Vector a, Matrix b)
        {
            if (b.Columns != a.Length)
                throw new Exception("not the same size!");

            var result = new Matrix(b.Rows, b.Columns);
            for (int i = 0; i < b.Rows; i++)
            {
                for (int j = 0; j < b.Columns; j++)
                {
                    result[i, j] = a[j] + b[i, j];
                }
            }

            return result;
        }

        public virtual Matrix Add(Matrix a, Matrix b)
        {
            CheckShape(a, b);

            var result = new double[a.Rows, a.Columns];
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Columns; j++)
                {
                    result[i, j] = a[i, j] + b[i, j];
                }
            }
            return new Matrix(result);
        }

        public virtual Tensor Add(Tensor a, Tensor b)
        {
            CheckShape(a, b);

            var result = a.GetSameShape();
            for (int i = 0; i < a.Depth; i++)
            {
                for (int j = 0; j < a.Rows; j++)
                {
                    for (int k = 0; k < a.Columns; k++)
                    {
                        result[i, j, k] = a[i, j, k] + b[i, j, k];
                    }
                }
            }
            return result;
        }

        #endregion


        #region Minus

        public Vector Minus(Vector v, double b)
        {
            var result = new double[v.Length];
            for (int i = 0; i < v.Length; i++)
            {
                result[i] = v[i] - b;
            }
            return new Vector(result);
        }

        public Vector Minus(double b, Vector v)
        {
            var result = new double[v.Length];
            for (int i = 0; i < v.Length; i++)
            {
                result[i] = b - v[i];
            }
            return new Vector(result);
        }

        public Vector Minus(Vector a, Vector b)
        {
            CheckShape(a, b);

            var result = new double[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                result[i] = a[i] - b[i];
            }
            return new Vector(result);
        }

        public virtual Matrix Minus(Matrix m, double b)
        {
            var result = new double[m.Rows, m.Columns];
            for (int i = 0; i < m.Rows; i++)
            {
                for (int j = 0; j < m.Columns; j++)
                {
                    result[i, j] = m[i, j] - b;
                }
            }
            return new Matrix(result);
        }

        public virtual Matrix Minus(double b, Matrix m)
        {
            var result = new double[m.Rows, m.Columns];
            for (int i = 0; i < m.Rows; i++)
            {
                for (int j = 0; j < m.Columns; j++)
                {
                    result[i, j] = b - m[i, j];
                }
            }
            return new Matrix(result);
        }

        public virtual Matrix Minus(Matrix a, Vector b)
        {
            if (a.Rows != b.Length)
                throw new Exception("not the same size!");

            var result = new Matrix(a.Rows, a.Columns);
            for (int i = 0; i < a.Columns; i++)
            {
                for (int j = 0; j < a.Rows; j++)
                {
                    result[j, i] = a[j, i] - b[j];
                }
            }

            return result;
        }

        public virtual Matrix Minus(Vector a, Matrix b)
        {
            if (b.Columns != a.Length)
                throw new Exception("not the same size!");

            var result = new Matrix(b.Rows, b.Columns);
            for (int i = 0; i < b.Rows; i++)
            {
                for (int j = 0; j < b.Columns; j++)
                {
                    result[i, j] = a[j] - b[i, j];
                }
            }

            return result;
        }

        public virtual Matrix Minus(Matrix a, Matrix b)
        {
            CheckShape(a, b);

            var result = new double[a.Rows, a.Columns];
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Columns; j++)
                {
                    result[i, j] = a[i, j] - b[i, j];
                }
            }
            return new Matrix(result);
        }

        #endregion


        #region Multiple

        public Vector Multiple(Vector v, double b)
        {
            var result = new double[v.Length];
            for (int i = 0; i < v.Length; i++)
            {
                result[i] = v[i] * b;
            }
            return new Vector(result);
        }

        public double Multiple(Vector a, Vector b)
        {
            CheckShape(a, b);

            return MultipleElementWise(a, b).Sum();
        }

        public virtual Matrix Multiple(Matrix m, double b)
        {
            var result = new double[m.Rows, m.Columns];
            for (int i = 0; i < m.Rows; i++)
            {
                for (int j = 0; j < m.Columns; j++)
                {
                    result[i, j] = m[i, j] * b;
                }
            }
            return new Matrix(result);
        }

        public virtual Matrix Multiple(Matrix a, Vector v)
        {
            var m = v.ToMatrix(true);
            return Multiple(a, m);
        }

        public virtual Matrix Multiple(Vector v, Matrix a)
        {
            var m = v.ToMatrix();
            return Multiple(m, a);
        }

        public virtual Matrix Multiple(Matrix a, Matrix b)
        {
            if (a.Columns != b.Rows)
                throw new Exception($"a.Columns={a.Columns} and b.Rows={b.Rows} are not equal!");

            var result = new double[a.Rows, b.Columns];
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < b.Columns; j++)
                {
                    result[i, j] = 0;
                    for (int k = 0; k < a.Columns; k++)
                    {
                        result[i, j] += (a[i, k] * b[k, j]);
                    }
                }
            }
            return new Matrix(result);
        }

        public Vector MultipleElementWise(Vector a, Vector b)
        {
            CheckShape(a, b);

            var result = new double[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                result[i] = a[i] * b[i];
            }
            return new Vector(result);
        }

        public virtual Matrix MultipleElementWise(Matrix a, Matrix b)
        {
            CheckShape(a, b);

            var result = a.GetSameShape();
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Columns; j++)
                {
                    result[i, j] = a[i, j] * b[i, j];
                }
            }
            return result;
        }

        public virtual Tensor MultipleElementWise(Tensor a, Tensor b)
        {
            CheckShape(a, b);

            var result = a.GetSameShape();
            for (int i = 0; i < a.Depth; i++)
            {
                for (int j = 0; j < a.Rows; j++)
                {
                    for (int k = 0; k < a.Columns; k++)
                    {
                        result[i, j, k] = a[i, j, k] * b[i, j, k];
                    }
                }
            }
            return result;
        }

        #endregion


        #region Divide

        public Vector Divide(Vector v, double b)
        {
            var result = new double[v.Length];
            for (int i = 0; i < v.Length; i++)
            {
                result[i] = v[i] / b;
            }
            return new Vector(result);
        }

        public Vector Divide(double b, Vector v)
        {
            var result = new double[v.Length];
            for (int i = 0; i < v.Length; i++)
            {
                result[i] = b / v[i];
            }
            return new Vector(result);
        }

        public virtual Matrix Divide(Matrix m, double b)
        {
            var result = new double[m.Rows, m.Columns];
            for (int i = 0; i < m.Rows; i++)
            {
                for (int j = 0; j < m.Columns; j++)
                {
                    result[i, j] = m[i, j] / b;
                }
            }
            return new Matrix(result);
        }

        public virtual Matrix Divide(double b, Matrix m)
        {
            var result = new double[m.Rows, m.Columns];
            for (int i = 0; i < m.Rows; i++)
            {
                for (int j = 0; j < m.Columns; j++)
                {
                    result[i, j] = b / m[i, j];
                }
            }
            return new Matrix(result);
        }

        #endregion

        public virtual Vector Apply(Vector v, Func<double,double> function)
        {
            var result = new Vector(v.Length);
            for (int i = 0; i < v.Length; i++)
            {
                result[i] = function(v[i]);
            }
            return result;
        }

        public virtual Matrix Apply(Matrix m, Func<double, double> function)
        {
            var result = new Matrix(m.Rows, m.Columns);
            for (int i = 0; i < m.Rows; i++)
            {
                for (int j = 0; j < m.Columns; j++)
                {
                    result[i, j] = function(m[i, j]);
                }
            }
            return result;
        }

        public virtual Tensor Apply(Tensor t, Func<double,double> function)
        {
            var result = t.GetSameShape();
            for (int i = 0; i < t.Depth; i++)
            {
                for (int j = 0; j < t.Rows; j++)
                {
                    for (int k = 0; k < t.Columns; k++)
                    {
                        t[i, j, k] = function(t[i, j, k]);
                    }
                }
            }
            return result;
        }

        public virtual double Convolute(Matrix input, Matrix filter, int startRow, int startColumn)
        {
            var result = 0d;
            for (int i = 0; i < filter.Rows; i++)
            {
                for (int j = 0; j < filter.Columns; j++)
                {
                    result += filter[i, j] * input[startRow + i, startColumn + j];
                }
            }
            return result;
        }

        public virtual double Convolute(Tensor input, Tensor filter, int startRow, int startColumn)
        {
            CheckTensorDepth(input, filter);

            var result = 0d;
            for (int i = 0; i < filter.Depth; i++)
            {
                for (int j = 0; j < input.Rows; j++)
                {
                    for (int k = 0; k < input.Columns; k++)
                    {
                        result += filter[i, j, k] * input[i, startRow + j, startColumn + k];
                    }
                }
            }
            return result;
        }

        #region Helper Functions

        public void CheckShape(Tensor a, Tensor b)
        {
            if (a.Rows != b.Rows ||
                a.Columns != b.Columns ||
                a.Depth != b.Depth)
                throw new Exception("tensors are not the same shape!");
        }

        public void CheckShape(Vector a, Vector b)
        {
            if (a.Length != b.Length)
                throw new Exception($"vector a.Length={a.Length} and b.Length={b.Length} are not equal!");
        }

        public void CheckShape(Matrix a, Matrix b)
        {
            if (a.Rows != b.Rows || a.Columns != b.Columns)
                throw new Exception($"matrix shape a:[{a.Rows},{a.Columns}] and b:[{b.Rows},{b.Columns}] are not equal!");
        }
        
        public void CheckTensorDepth(Tensor a, Tensor b)
        {
            if (a.Depth != b.Depth)
                throw new Exception("Not have the same depth!");
        }

        #endregion
    }
}
