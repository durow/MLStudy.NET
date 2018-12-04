using MLStudy;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;

namespace PlayGround.Plays
{
    public class Performance : IPlay
    {
        public void Play()
        {
            var data = DataEmulator.Instance.RandomArray(200000);

            var t1 = new TensorOld(data, 500, 400);
            var t2 = new TensorOld(data, 400, 500);
            var tResult = new TensorOld(new double[250000], 500, 500);

            var a1 = new double[500, 400];
            var a2 = new double[400, 500];
            var result = new double[500, 500];
            var rand = new Random();
            for (int i = 0; i < 500; i++)
            {
                for (int j = 0; j < 400; j++)
                {
                    a1[i, j] = rand.NextDouble();
                    a2[j, i] = rand.NextDouble();
                }
            }


            var np = new NumSharp.Core.NumPy<double>();
            var arr1 = np.array(data).reshape(500, 400);
            var arr2 = np.array(data).reshape(400, 500);
            var arr3 = np.array(new double[250000]).reshape(500, 500);
            var arrResult = np.array(new double[250000]).reshape(500, 500);

            var fast1 = Array.CreateInstance(typeof(double), 500, 400);
            var fast2 = Array.CreateInstance(typeof(double), 400, 500);
            var fastResult = Array.CreateInstance(typeof(double), 500, 500);
            for (int i = 0; i < 500; i++)
            {
                for (int j = 0; j < 400; j++)
                {
                    fast1.SetValue(rand.NextDouble(), i, j);
                    fast2.SetValue(rand.NextDouble(), j, i);
                }
            }

            var sum = 0d;
            var start1 = DateTime.Now;
            for (int i = 0; i < 500; i++)
            {
                for (int j = 0; j < 500; j++)
                {
                    sum = 0d;
                    for (int k = 0; k < 400; k++)
                    {
                        sum += a1[i, k] * a2[k, j];
                    }
                    result[i, j] = sum;
                }
            }

            var start3 = DateTime.Now;
            for (int i = 0; i < 500; i++)
            {
                for (int j = 0; j < 500; j++)
                {
                    sum = 0d;
                    for (int k = 0; k < 400; k++)
                    {
                        sum += arr1[i, k] * arr2[k, j];
                    }
                    arrResult[i, j] = sum;
                }
            }
            var ts3 = DateTime.Now - start3;
            Console.WriteLine($"NDArry Time:{ts3.TotalMilliseconds} ms");

            var start4 = DateTime.Now;
            //for (int i = 0; i < 500; i++)
            //{
            //    for (int j = 0; j < 500; j++)
            //    {
            //        sum = 0d;
            //        for (int k = 0; k < 400; k++)
            //        {
            //            sum += t1.GetValueFast(i, k) * t2.GetValueFast(k, j);
            //        }
            //        tResult.SetValueFast(sum, i, j);
            //    }
            //}
            TensorOld.Multiple(t1, t2, tResult);
            var ts4 = DateTime.Now - start4;
            Console.WriteLine($"Tensor Time:{ts4.TotalMilliseconds} ms");

            var tens = new TensTest();
            var start5 = DateTime.Now;
            tens.Multi();
            var ts5 = DateTime.Now - start5;
            Console.WriteLine($"Tens Time:{ts5.TotalMilliseconds} ms");
        }
    }

    public class TensTest
    {
        public double[,] a1 { get; set; }
        public double[,] a2 { get; set; }
        public double[,] result { get; set; }

        public double this[int index]
        {
            get
            {
                return 0.123;
            }
        }
        public TensTest()
        {
            a1 = new double[500, 400];
            a2 = new double[400, 500];
            result = new double[500, 500];
            var rand = new Random();
            for (int i = 0; i < 500; i++)
            {
                for (int j = 0; j < 400; j++)
                {
                    a1[i, j] = rand.NextDouble();
                    a2[j, i] = rand.NextDouble();
                }
            }

            result = new double[500, 500];
        }

        public void Multi()
        {
            var sum = 0d;
            for (int i = 0; i < 500; i++)
            {
                for (int j = 0; j < 500; j++)
                {
                    for (int k = 0; k < 400; k++)
                    {
                        sum += a1[i, k] * a2[k, j];
                    }
                    result[i, j] = sum;
                }
            }
        }

    }
}
