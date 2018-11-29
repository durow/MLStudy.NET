using System;
using System.Collections.Generic;
using System.Text;

namespace PlayGround.Plays
{
    public class Test : IPlay
    {
        public void Play()
        {
            var dict = new Dictionary<int[], int>();
            var a1 = new int[] { 1, 2 };
            var a2 = new int[] { 1, 2 };

            dict.Add(a1, 12);
            dict.Add(a2, 45);

            Console.WriteLine(dict[a2]);
        }
    }
}
