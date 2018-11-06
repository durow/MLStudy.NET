/*
 * Description:4 dimension tensor
 * Author:YunXiao An
 * Date:2018.11.05
 */

using System;

namespace MLStudy
{
    public sealed class Tensor4
    {
        private double[,,,] values;
        public int Dimension => 4;
        public int D1 { get { return values.GetLength(0); } }
        public int D2 { get { return values.GetLength(1); } }
        public int D3 { get { return values.GetLength(2); } }
        public int D4 { get { return values.GetLength(3); } }


        public Tensor4(int d1, int d2, int d3, int d4)
        {
            values = new double[d1, d2, d3, d4];
        }

        public double this[int d1, int d2, int d3, int d4]
        {
            get
            {
                return values[d1, d2, d3, d4];
            }
            set
            {
                values[d1, d2, d3, d4] = value;
            }
        }

        public Tensor4(double[,,,] data)
        {
            values = data;
        }

        public Tensor4 GetSameShape()
        {
            return new Tensor4(D1, D2, D3, D4);
        }

        public Tensor3 this[int d1Index]
        {
            get
            {
                return Get(d1Index);
            }
        }

        public Tensor3 Get(int d1Index)
        {
            var result = new Tensor3(D2, D3, D4);
            for (int i = 0; i < D2; i++)
            {
                for (int j = 0; j < D3; j++)
                {
                    for (int k = 0; k < D4; k++)
                    {
                        result[i, j, k] = values[d1Index, i, j, k];
                    }
                }
            }
            return result;
        }

        public Tensor4 Apply(Func<double,double> function)
        {
            var result = GetSameShape();
            for (int i = 0; i < D1; i++)
            {
                for (int j = 0; j < D2; j++)
                {
                    for (int k = 0; k < D3; k++)
                    {
                        for (int l = 0; l < D4; l++)
                        {
                            result[i, j, k, l] = values[i, j, k, l];
                        }
                    }
                }
            }
            return result;
        }

        public static Tensor4 operator +(Tensor4 t, double b)
        {
            return t.Apply(d => d + b);
        }

        public static Tensor4 operator +(double b, Tensor4 t)
        {
            return t + b;
        }

        public static Tensor4 operator -(Tensor4 t, double b)
        {
            return t.Apply(d => d - b);
        }

        public static Tensor4 operator -(double b, Tensor4 t)
        {
            return t.Apply(d => b - d);
        }

        public static Tensor4 operator *(Tensor4 t, double b)
        {
            return t.Apply(d => d * b);
        }

        public static Tensor4 operator *(double b, Tensor4 t)
        {
            return t * b;
        }

        public static Tensor4 operator /(Tensor4 t, double b)
        {
            return t.Apply(d => d / b);
        }

        public static Tensor4 operator /(double b, Tensor4 t)
        {
            return t.Apply(d => b / d);
        }
    }
}
