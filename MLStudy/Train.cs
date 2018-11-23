using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class Train
    {
        public bool Training { get; private set; }
        public IEngine Engine { get; private set; }
        public int BatchSize { get; set; }
        public int Epoch { get; set; }
        public bool RandomBatch { get; set; }
        public int Seed { get; set; }

        public Tensor TrainX { get; private set; }
        public Tensor TrainY { get; private set; }
        public Tensor TestX { get; private set; }
        public Tensor TestY { get; private set; }

        private Tensor xBuff;
        private Tensor yBuff;
        private Random rand { get; set; }
        private int pointer = 0;
        private int batchPerEpoch;
        private int sampleCount;
        private int xWidth;
        private int yWidth;
        private int batchCounter = 1;
        private int epochCounter = 0;
        private DateTime startTime;

        public event EventHandler BatchComplete;

        public Train(IEngine engine, int batchSize, int epoch, bool randomBatch = false)
        {
            Engine = engine;
            BatchSize = batchSize;
            Epoch = epoch;
            RandomBatch = randomBatch;
        }

        public void StartTrain(Tensor trainX, Tensor trainY, Tensor testX, Tensor testY)
        {
            TrainX = trainX;
            TrainY = trainY;
            TestX = testX;
            TestY = testY;

            xBuff = new Tensor(BatchSize, trainX.shape[1]);
            yBuff = new Tensor(BatchSize, trainY.shape[1]);
            sampleCount = TrainX.shape[0];
            xWidth = TrainX.shape[1];
            yWidth = TrainY.shape[1];

            batchPerEpoch = trainX.shape[0] / BatchSize;
            if (trainX.shape[0] % BatchSize != 0)
                batchPerEpoch++;

            if (RandomBatch)
                rand = new Random(Seed);

            Training = true;
            epochCounter = 0;
            batchCounter = 1;
            startTime = DateTime.Now;
            while (Training && epochCounter < Epoch)
            {
                Console.WriteLine($"============== Epoch {epochCounter + 1} ==============");
                var epochStart = DateTime.Now;

                TrainEpoch();

                var trainLoss = Engine.GetTrainLoss();
                var trainAcc = Engine.GetTrainAccuracy();

                ShowTime(epochStart);
                ShowTrainLoss(trainLoss, trainAcc);

                var testLoss = 0d;
                var testAcc = 0d;
                if (TestX != null && TestY != null)
                {
                    var testYHat = Engine.Predict(TestX);
                    testLoss = Engine.GetLoss(TestY, testYHat);
                    testAcc = Engine.GetAccuracy(testY, testYHat);
                    ShowTestLoss(testLoss, testAcc);
                }

                epochCounter++;
            }
            ShowCompelteInfo();
        }

        public void Stop()
        {
            Training = false;
        }

        private void TrainEpoch()
        {
            var counter = 0;
            while (Training && counter < batchPerEpoch)
            {
                SetBatchData();
                Engine.Step(xBuff, yBuff);
                counter++;
                batchCounter++;
            }
        }

        private void SetBatchData()
        {
            if (RandomBatch)
                SetBatchDataRandom();
            else
            {
                for (int i = 0; i < BatchSize; i++)
                {
                    var index = pointer % TrainX.shape[0];
                    CopyToBatch(index, i);
                    pointer++;
                }
            }
        }

        private void SetBatchDataRandom()
        {
            var selected = GetRandomDistinct(0, sampleCount, BatchSize, Seed);
            for (int i = 0; i < selected.Count; i++)
            {
                CopyToBatch(selected[i], i);
            }
        }

        private void ShowTime(DateTime epochStart)
        {
            var cost = DateTime.Now - epochStart;
            var passed = DateTime.Now - startTime;
            var remain = GetRemainTimeSpan(passed);
            var estimate = DateTime.Now + remain;

            Console.WriteLine($"Time>>cost:{cost.TotalSeconds.ToString("F2")}s  passed:{GetTimeSpanFormat(passed)} remain:{GetTimeSpanFormat(remain)}  ETA:{estimate.ToString("yyyy-MM-dd HH:mm:ss")}");
        }

        private void ShowTrainLoss(double trainLoss, double trainAccuracy)
        {

            Console.WriteLine($"Train>>Loss:{trainLoss}\tAccuracy:{trainAccuracy}");
        }

        private void ShowTestLoss(double testLoss, double testAccuracy)
        {
            Console.WriteLine($"Test>>Loss:{testLoss}\tAccuracy:{testAccuracy}");
        }

        private void ShowCompelteInfo()
        {
            var ts = DateTime.Now - startTime;
            Console.WriteLine($"Train complete, use total time:{GetTimeSpanFormat(ts)}");
        }

        private void CopyToBatch(int trainIndex, int batchIndex)
        {
            Array.Copy(TrainX.GetRawValues(), trainIndex * xWidth, xBuff.GetRawValues(), batchIndex * xWidth, xWidth);
            Array.Copy(TrainY.GetRawValues(), trainIndex * yWidth, yBuff.GetRawValues(), batchIndex * yWidth, yWidth);
        }

        public static List<int> GetRandomDistinct(int min, int max, int count, int seed)
        {
            if (count < 1)
                return new List<int>();

            if (count > (max - min))
                throw new Exception("count must < max-min");

            if (count > (max - min) / 2)
            {
                var temp = GetRandomDistinct(min, max, max - min - count, seed);
                var remain = new List<int>();
                for (int i = min; i < max; i++)
                {
                    if (temp.Contains(i))
                        continue;
                    remain.Add(i);
                }
                return remain;
            }

            var rand = new Random(seed);
            var result = new List<int>();
            while (result.Count < count)
            {
                var r = rand.Next(min, max);
                if (!result.Contains(r))
                    result.Add(r);
            }
            return result;
        }

        private string GetTimeSpanFormat(TimeSpan ts)
        {
            var start = false;
            var result = new List<string>();
            if (ts.Days > 0 || start)
            {
                result.Add($"{ts.Days}d");
                start = true;
            }
            if (ts.Hours > 0 || start)
            {
                result.Add($"{ts.Hours}h");
                start = true;

            }
            if (ts.Minutes > 0 || start)
            {
                result.Add($"{ts.Minutes}m");

            }
            result.Add($"{ts.Seconds}s");

            return string.Join("", result);
        }

        private TimeSpan GetRemainTimeSpan(TimeSpan passed)
        {
            var ms = passed.TotalMilliseconds * (Epoch - epochCounter + 1) / (epochCounter + 1);
            return TimeSpan.FromMilliseconds(ms);
        }
    }
}
