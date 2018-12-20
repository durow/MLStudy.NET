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

            var test = MLStudy.Num.Tensor.Fill(1.2f, 2, 3);
            Console.WriteLine(test.GetType().Name);

            Console.ReadKey();
        }
    }
}