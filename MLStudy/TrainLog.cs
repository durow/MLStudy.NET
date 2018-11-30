using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class TrainLog
    {
        public DateTime Time { get; set; }
        public string Mission { get; set; }
        public string ModelType { get; set; }
        public string PreProcessor { get; set; }
        public string LabelCodec { get; set; }
        public string Normalizer { get; set; }
        public int BatchSize { get; set; }
        public int Epoch { get; set; }
        public bool RandomBatch { get; set; }

        public List<PerformanceLog> PerformanceLogs { get; set; }

        public void Clear()
        {
            PerformanceLogs.Clear();
        }
    }

    public class PerformanceLog
    {
        public DateTime Time { get; set; }
        public double TrainLoss { get; set; }
        public double TestLoss { get; set; }
        public double TrainAccuracy { get; set; }
        public double TestAccuracy { get; set; }
    }
}
