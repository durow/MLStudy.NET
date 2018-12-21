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

            var test = MLStudy.Num.Tensor.Empty(2, 3);
            var test2 = test.ReShape(3, 2);
            Console.WriteLine(test);
            Console.WriteLine(test2);

            Console.ReadKey();
        }
    }
}