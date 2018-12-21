using MLStudy;
using PlayGround.Plays;
using System;

namespace PlayGround
{
    class Program
    {
        static void Main(string[] args)
        {
            //IPlay play = new MNIST();
            //play.Play();
            var test = MLStudy.Num.Tensor.Ones(3, 4);
            Console.WriteLine(test.DataType);

            Console.ReadKey();
        }
    }
}