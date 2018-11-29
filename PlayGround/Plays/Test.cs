using MLStudy;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;
using MLStudy.Playground;

namespace PlayGround.Plays
{
    public class Test : IPlay
    {
        public void Play()
        {
            var data = DataEmulator.Instance.RandomArray(20000);

            var t1 = new Tensor(data, 200, 100);
            var t2 = new Tensor(data, 100, 200);
            var tResult = new Tensor(new double[40000],200,200);

            var a1 = new double[200, 100];
            var a2 = new double[100, 200];
            var result = new double[200, 200];
            var rand = new Random();
            for (int i = 0; i < 200; i++)
            {
                for (int j = 0; j < 100; j++)
                {
                    a1[i, j] = rand.NextDouble();
                    a2[j, i] = rand.NextDouble();
                }
            }

            var tt1 = new TensorTest(data,200,100);
            var tt2 = new TensorTest(data,100,200);
            var ttResult = new TensorTest(new double[40000], 200, 200);


            var np = new NumSharp.Core.NumPy<double>();
            var arr1 = np.array(data).reshape(200, 100);
            var arr2 = np.array(data).reshape(100, 200);
            var arr3 = np.array(new double[40000]).reshape(200, 200);
            var arrResult = np.array(new double[40000]).reshape(200,200);

            var fast1 = Array.CreateInstance(typeof(double), 200, 100);
            var fast2 = Array.CreateInstance(typeof(double), 100, 200);
            var fastResult = Array.CreateInstance(typeof(double), 200, 200);
            for (int i = 0; i < 200; i++)
            {
                for (int j = 0; j < 100; j++)
                {
                    fast1.SetValue(rand.NextDouble(),i,j);
                    fast2.SetValue(rand.NextDouble(),j,i);
                }
            }

            var sum = 0d;
            var start1 = DateTime.Now;
            for (int i = 0; i < 200; i++)
            {
                for (int j = 0; j < 200; j++)
                {
                    sum = 0d;
                    for (int k = 0; k < 100; k++)
                    {
                        sum += a1[i, k] * a2[k, j];
                    }
                    result[i, j] = sum;
                }
            }
            var ts1 = DateTime.Now - start1;
            Console.WriteLine($"Array Time:{ts1.TotalMilliseconds} ms");


            var start2 = DateTime.Now;
            for (int i = 0; i < 200; i++)
            {
                for (int j = 0; j < 200; j++)
                {
                    sum = 0d;
                    for (int k = 0; k < 100; k++)
                    {
                        sum += tt1[i, k] * tt2[k, j];
                    }
                    ttResult[i, j] = sum;
                }
            }
            //Tensor.Multiple(t1, t2, tResult);
            var ts2 = DateTime.Now - start2;
            Console.WriteLine($"TensorTest Time:{ts2.TotalMilliseconds} ms");


            var start3 = DateTime.Now;
            for (int i = 0; i < 200; i++)
            {
                for (int j = 0; j < 200; j++)
                {
                    sum = 0d;
                    for (int k = 0; k < 100; k++)
                    {
                        sum += arr1[i, k] * arr2[k, j];
                    }
                    arrResult[i, j] = sum;
                }
            }
            var ts3 = DateTime.Now - start3;
            Console.WriteLine($"NDArry Time:{ts3.TotalMilliseconds} ms");

            var start4 = DateTime.Now;
            for (int i = 0; i < 200; i++)
            {
                for (int j = 0; j < 200; j++)
                {
                    sum = 0d;
                    for (int k = 0; k < 100; k++)
                    {
                        sum += t1[i, k] * t2[k, j];
                    }
                    tResult[i, j] = sum;
                }
            }
            var ts4 = DateTime.Now - start4;
            Console.WriteLine($"Tensor Time:{ts4.TotalMilliseconds} ms");

            var start5 = DateTime.Now;
            for (int i = 0; i < 200; i++)
            {
                for (int j = 0; j < 200; j++)
                {
                    sum = 0d;
                    for (int k = 0; k < 100; k++)
                    {
                        sum += (double)fast1.GetValue(i,k) * (double)fast2.GetValue(k,j);
                    }
                    fastResult.SetValue(sum,i,j);
                }
            }
            var ts5 = DateTime.Now - start5;
            Console.WriteLine($"TensorFast Time:{ts5.TotalMilliseconds} ms");
        }
    }

    public class TensTest
    {
        public ArraySegment<double> values;

    }
}
