using PlayGround.Plays;
using System;

namespace PlayGround
{
    class Program
    {
        static void Main(string[] args)
        {
            IPlay play = new Performance();
            play.Play();

            Console.ReadKey();
        }
    }
}
