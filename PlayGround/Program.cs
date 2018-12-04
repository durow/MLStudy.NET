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

            var test = Tensor.Ones(2, 3, 2);
            var a = test[1].ReShape(2, 3);
            a.Values[0, 0] = 321;
            var b = a + 10;
            Console.WriteLine(test);

            Console.ReadKey();
        }
    }
}