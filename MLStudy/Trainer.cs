using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class Trainer
    {
        public bool Training { get; private set; }
        public IModel Model { get; private set; }
        public int BatchSize { get; set; }
        public int Epoch { get; set; }
        public bool RandomBatch { get; set; }
        public int CounterLimit { get; set; } = 10000;

        public Tensor TrainX { get; private set; }
        public Tensor TrainY { get; private set; }
        public Tensor TestX { get; private set; }
        public Tensor TestY { get; private set; }

        private Tensor xBuff;
        private Tensor yBuff;
        private int pointer = 0;
        private int batchPerEpoch;
        private int sampleCount;
        private int xWidth;
        private int yWidth;
        private int batchCounter = 1;
        private int epochCounter = 0;
        private DateTime startTime;

        public event EventHandler BatchComplete;

        public Trainer(IModel model, int batchSize, int epoch, bool randomBatch = false)
        {
            Model = model;
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

            Training = true;
            epochCounter = 0;
            batchCounter = 1;
            startTime = DateTime.Now;
            while (Training && epochCounter < Epoch)
            {
                Console.WriteLine($"============== Epoch {epochCounter + 1} ==============");
                var epochStart = DateTime.Now;
                TrainEpoch();
                ShowTime(epochStart);
                //var trainLoss = Model.GetTrainLoss();
                //var trainAcc = Model.GetTrainAccuracy();

                //ShowTrainLoss(trainLoss, trainAcc);

                var testLoss = 0d;
                var testAcc = 0d;
                if (TestX != null && TestY != null)
                {
                    var testYHat = Model.Predict(TestX);
                    testLoss = Model.GetLoss(TestY, testYHat);
                    testAcc = Model.GetAccuracy(TestY, testYHat);
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
            var counter = 1;
            while (Training && counter <= batchPerEpoch)
            {
                SetBatchData();

                Model.Step(xBuff, yBuff);

                var trainLoss = Model.GetTrainLoss();
                var trainAcc = Model.GetTrainAccuracy();

                ShowTrainLoss(trainLoss, trainAcc, counter);

                counter++;
                batchCounter++;
            }
            Console.WriteLine();
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
            var selected = Utilities.GetRandomDistinct(0, sampleCount, BatchSize);
            for (int i = 0; i < selected.Length; i++)
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

        private void ShowTrainLoss(double trainLoss, double trainAccuracy, int counter)
        {

            Console.WriteLine($"Epoch:{epochCounter+1}>>Batch:{counter*BatchSize}/{sampleCount}>>Loss:{trainLoss.ToString("F4")}  Accuracy:{trainAccuracy.ToString("F4")}");
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
