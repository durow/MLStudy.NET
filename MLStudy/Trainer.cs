using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
using System.Linq;
using System.Text;

namespace MLStudy
{
    public class Trainer
    {
        public string Mission { get; set; }
        public string SaveDir { get; set; }
        public bool Training { get; private set; }
        public IPreProcessor PreProcessor { get; set; }
        public DiscreteCodec LabelCodec { get; set; }
        public INormalizer Normalizer { get; set; }
        public int BatchSize { get; set; }
        public int Epoch { get; set; }
        public bool RandomBatch { get; set; }
        public int PrintSteps { get; set; } = 10;
        public IModel Model { get; private set; }

        public int SaveLogSteps { get; set; } = 0;
        public int SaveTrainerEpochs { get; set; } = 0;

        public Tensor TrainX { get; private set; }
        public Tensor TrainY { get; private set; }
        public Tensor TestX { get; private set; }
        public Tensor TestY { get; private set; }

        public double LastTrainLoss { get; internal set; }
        public double LastTrainAccuracy { get; internal set; }
        public double LastTestLoss { get; internal set; }
        public double LastTestAccuracy { get; internal set; }
        public TrainLog Log { get; private set; }

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
        public event EventHandler StepComplete;

        public Trainer(IModel model, int batchSize, int epoch, bool randomBatch = false)
        {
            Model = model;
            BatchSize = batchSize;
            Epoch = epoch;
            RandomBatch = randomBatch;
        }

        public void StartTrain(DataTable trainX, DataTable trainY, DataTable testX, DataTable testY)
        {
            if (PreProcessor == null)
                throw new Exception("you need a PreProcessor!");

            StartTrain(
                PreProcessor.PreProcessX(trainX),
                PreProcessor.PreProcessY(trainY),
                PreProcessor.PreProcessX(testX),
                PreProcessor.PreProcessY(testY));
        }

        public void StartTrain(Tensor trainX, Tensor trainY, Tensor testX, Tensor testY)
        {
            Console.WriteLine("Preparing data...");
            SetTrainingData(trainX, trainY, testX, testY);
            Console.WriteLine("Complete!");

            Training = true;
            epochCounter = 0;
            batchCounter = 1;
            startTime = DateTime.Now;
            var lastSave = false;

            while (Training && epochCounter < Epoch)
            {
                Console.WriteLine($"\n==================== Epoch {epochCounter + 1} ====================\n");
                var epochStart = DateTime.Now;
                TrainEpoch();
                ShowTime(epochStart);

                var trainYHat = Model.Predict(TrainX);
                LastTrainLoss = Model.GetLoss(TrainY, trainYHat);
                LastTrainAccuracy = Model.GetAccuracy(TrainY, trainYHat);
                ShowTrainLossAll();

                //test the test data when one Epoch complete
                if (TestX != null && TestY != null)
                {
                    var testYHat = Model.Predict(TestX);
                    LastTestLoss = Model.GetLoss(TestY, testYHat);
                    LastTestAccuracy = Model.GetAccuracy(TestY, testYHat);
                    ShowTestLoss();
                }

                if (SaveTrainerEpochs > 0 && epochCounter % SaveTrainerEpochs == 0)
                {
                    SaveTrainer();
                    if (Epoch - epochCounter == 1)
                        lastSave = true;
                }

                epochCounter++;
            }

            if (SaveTrainerEpochs > 0 && !lastSave)
                SaveTrainer();

            ShowCompelteInfo();
        }

        public ClassificationMachine GetClassificationMachine()
        {
            return new ClassificationMachine(Model)
            {
                LabelCodec = LabelCodec,
                Normalizer = Normalizer,
                PreProcessor = PreProcessor,
            };
        }

        public RegressionMachine GetRegressionMachine()
        {
            return new RegressionMachine(Model)
            {
                Normalizer = Normalizer,
                PreProcessor = PreProcessor,
            };
        }

        public void SaveTrainer()
        {
            if (!Directory.Exists(SaveDir))
                Directory.CreateDirectory(SaveDir);

            var time = DateTime.Now.ToString("yyyy_MM_dd_HH_mm_ss");
            var filename = $"{SaveDir}\\{Mission}_Trainer_{time}.xml";
            Storage.Save(this, filename);
        }

        private void SetTrainingData(Tensor trainX, Tensor trainY, Tensor testX, Tensor testY)
        {
            if (Normalizer != null)
            {
                TrainX = Normalizer.Normalize(trainX);
                if (testX != null)
                    TestX = Normalizer.Normalize(testX);
            }
            else
            {
                TrainX = trainX;
                TestX = testX;
            }
            if (LabelCodec != null)
            {
                TrainY = LabelCodec.Encode(trainY.GetRawValues());
                if (testY != null)
                    TestY = LabelCodec.Encode(testY.GetRawValues());
            }
            else
            {
                TrainY = trainY;
                TestY = testY;
            }

            var shape = TrainX.Shape;
            shape[0] = BatchSize;
            xBuff = new Tensor(shape);
            yBuff = new Tensor(BatchSize, TrainY.shape[1]);

            sampleCount = TrainX.shape[0];
            xWidth = TrainX.ElementCount / TrainX.shape[0];
            yWidth = TrainY.ElementCount / TrainY.shape[0];

            batchPerEpoch = trainX.shape[0] / BatchSize;
            if (trainX.shape[0] % BatchSize != 0)
                batchPerEpoch++;
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

                if (counter % PrintSteps == 0)
                {
                    LastTrainLoss = Model.GetTrainLoss();
                    LastTrainAccuracy = Model.GetTrainAccuracy();
                    ShowTrainLoss(counter);
                }

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

            Console.WriteLine($"\nTime>>cost:{cost.TotalSeconds.ToString("F2")}s  passed:{GetTimeSpanFormat(passed)} remain:{GetTimeSpanFormat(remain)}  ETA:{estimate.ToString("yyyy-MM-dd HH:mm:ss")}");
        }

        private void ShowTrainLoss(int counter)
        {
            Console.WriteLine($"Epoch:{epochCounter+1}>>Batch:{counter*BatchSize}/{sampleCount}>>Train>>Loss:{LastTrainLoss.ToString("F4")}  Accuracy:{LastTrainAccuracy.ToString("F4")}");
        }

        private void ShowTrainLossAll()
        {
            Console.WriteLine($"Epoch:{epochCounter + 1}>>Train>>Loss:{LastTrainLoss.ToString("F4")}\tAccuracy:{LastTrainAccuracy.ToString("F4")}");
        }

        private void ShowTestLoss()
        {
            Console.WriteLine($"Epoch:{epochCounter+1}>>Test>>Loss:{LastTestLoss.ToString("F4")}\tAccuracy:{LastTestAccuracy.ToString("F4")}");
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
