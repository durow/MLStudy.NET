﻿using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MLStudy
{
    public struct Vector
    {
        private double[] values;

        public int Length { get { return values.Length; } }

        public double this[int index]
        {
            get
            {
                return values[index];
            }
            set
            {
                values[index] = value;
            }
        }

        public Vector(int length)
        {
            values = new double[length];
        }

        public Vector(params double[] values)
        {
            this.values = values;
        }

        public double Sum()
        {
            return values.Sum();
        }

        public double Mean()
        {
            return values.Average();
        }

        public Matrix ToMatrix(bool verticalVector = false)
        {
            double[,] result;

            if(verticalVector)
            {
                result = new double[Length, 1];
                for (int i = 0; i < Length; i++)
                {
                    result[i, 0] = values[i];
                }
            }
            else
            {
                result = new double[1, Length];
                for (int i = 0; i < Length; i++)
                {
                    result[0, i] = values[i];
                }
            }

            return new Matrix(result);
        }

        public override string ToString()
        {
            var result = string.Join(",", values);
            return $"[{result}]";
        }

        public override bool Equals(object o)
        {
            if (!(o is Vector v))
                return false;

            if (Length != v.Length)
                return false;

            for (int i = 0; i < Length; i++)
            {
                if (this[i] != v[i])
                    return false;
            }

            return true;
        }

        #region Operations

        public static Vector operator +(Vector v, double b)
        {
            return Tensor.Add(v, b);
        }

        public static Vector operator +(double b, Vector v)
        {
            return Tensor.Add(v, b);
        }

        public static Vector operator +(Vector a, Vector b)
        {
            return Tensor.Add(a, b);
        }

        public static Vector operator *(Vector v, double b)
        {
            return Tensor.Multiple(v, b);
        }

        public static Vector operator *(double b, Vector v)
        {
            return Tensor.Multiple(v, b);
        }

        public static Vector operator *(Vector a, Vector b)
        {
            return Tensor.Multiple(a, b);
        }

        #endregion
    }
}
