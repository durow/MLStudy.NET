using MLStudy;
using MLStudy.Deep;
using System;
using System.Collections.Generic;
using System.Text;

namespace PlayGround.Plays
{
    public class ApplyTest : IPlay
    {
        public void Play()
        {
            var data = DataEmulator.Instance.RandomArray(100000);
            var tensor = new TensorOld(data, 500, 200);
            var full = new FullLayer(100);
            full.PrepareTrain(tensor);
            var start1 = DateTime.Now;
            var output = full.Forward(tensor);
            var ts1 = DateTime.Now - start1;
            Console.WriteLine($"Forward Time:{ts1.TotalMilliseconds}");
            
        }
    }
}
