using System;

namespace PlayGround
{
    class Program
    {
        static void Main(string[] args)
        {
            IPlay play = new XOR();
            play.Play();

            Console.ReadKey();
        }
    }
}
