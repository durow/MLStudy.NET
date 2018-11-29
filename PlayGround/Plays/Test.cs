using MLStudy;
using MLStudy.Maths;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;

namespace PlayGround.Plays
{
    public class Test : IPlay
    {
        public unsafe void Play()
        {
            var data = DataEmulator.Instance.RandomArray(20000);

            var t1 = new Tensor(data, 200, 100);
            var t2 = new Tensor(data, 100, 200);

            var a1 = new double[200, 100];
            var a2 = new double[100, 200];

            var tt1 = new TensorFast(data,200,100);
            var tt2 = new TensorFast(data,100,200);

            var result = new double[200, 200];
            var tResult = new Tensor(result);
            var ttResult = new TensorFast(result);

            var np = new NumSharp.Core.NumPy<double>();
            var arr1 = np.array(data).reshape(200, 100);
            var arr2 = np.array(data).reshape(100, 200);
            var arr3 = new NumSharp.Core.NDArray()
            var arrResult = np.array(new double[40000]).reshape(200,200);

            fixed (double* p=&data[0], p1 = &a1[0,0], p2=&a2[0,0])
            {
                for (int i = 0; i < data.Length; i++)
                {
                    p1[i] = p[i];
                    p2[i] = p[i];
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
            Console.WriteLine($"Time:{ts1.TotalMilliseconds} ms");


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
            Console.WriteLine($"Time:{ts2.TotalMilliseconds} ms");


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
            Console.WriteLine($"Time:{ts3.TotalMilliseconds} ms");

            
        }
    }

    public class TensTest
    {
        public ArraySegment<double> values;

    }
}
