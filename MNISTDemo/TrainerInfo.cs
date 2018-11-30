using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;

namespace MNISTDemo
{
    public class TrainerInfo : BindingBase
    {
        private bool isUing = false;

        public bool IsUsing
        {
            get { return isUing; }
            set
            {
                if (isUing != value)
                {
                    isUing = value;
                    RaisePropertyChanged("IsUsing");
                }
            }
        }

        public string FileName { get; set; }
        public string Mission { get; set; }
        public string Model { get; set; }
        public string TrainLoss { get; set; }
        public string TrainAccuracy { get; set; }
        public string TestLoss { get; set; } = "0";
        public string TestAccuracy { get; set; } = "0";
        public string Normalizer { get; set; }
        public string LabelCodec { get; set; }

        public static IEnumerable<TrainerInfo> ReadFromDir(string dir)
        {
            var files = Directory.GetFiles(dir);

            foreach (var file in files)
            {
                yield return ReadTrainerInfo(file);
            }
        }

        private static TrainerInfo ReadTrainerInfo(string file)
        {
            var doc = new XmlDocument();
            doc.Load(file);
            var root = doc.ChildNodes[1];
            return new TrainerInfo
            {
                FileName = file,
                Mission = root.SelectSingleNode("Mission").InnerText,
                Model = root.SelectSingleNode("Model").FirstChild.Name,
                TrainLoss = root.SelectSingleNode("LastTrainLoss").InnerText,
                TrainAccuracy = root.SelectSingleNode("LastTrainAccuracy").InnerText,
                TestLoss = root.SelectSingleNode("LastTestLoss").InnerText,
                TestAccuracy = root.SelectSingleNode("LastTestAccuracy").InnerText,
                Normalizer = root.SelectSingleNode("Normalizer").FirstChild.Name,
                LabelCodec = root.SelectSingleNode("LabelCodec").FirstChild.Name,
            };
        }
    }
}
