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

            var np = new NumSharp.Core.NumPy<float>();
            var a = np.array(new float[] { 1,2,3,4,5,6}).reshape(2, 3);
            Console.ReadKey();
        }
    }
}