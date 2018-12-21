using PlayGround.Plays;
using System;
using System.Data;

namespace PlayGround
{
    class Program
    {
        static void Main(string[] args)
        {
            //IPlay play = new MNIST();
            //play.Play();

            var a = MLStudy.Num.Tensor.Values(new int[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } }).AsType<double>();
            Console.WriteLine(a);
            Console.WriteLine(a.DataType);
            Console.WriteLine(a + 100);
            Console.ReadKey();
        }
    }
}