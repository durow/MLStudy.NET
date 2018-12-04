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

            var a = new FloatTensor(2, 3);
            var b = a + 0.5f;
            Console.ReadKey();
        }
    }
}
