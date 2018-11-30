using MLStudy.Abstraction;
using System;
using System.Collections.Generic;
using System.Text;
using System.Xml;

namespace MLStudy.Storages.Xml
{
    public class XmlTrainerStorage : XmlStorageBase<Trainer>
    {
        public override XmlElement Save(XmlDocument doc, object o)
        {
            if (o == null)
                return null;

            if (!(o is Trainer trainer))
                throw new Exception("Codec must be Trainer!");

            var el = GetRootElement(doc);
            XmlStorage.AddChild(el, "Mission", trainer.Mission);
            XmlStorage.AddChild(el, "BatchSize", trainer.BatchSize);
            XmlStorage.AddChild(el, "Epoch", trainer.Epoch);
            XmlStorage.AddChild(el, "RandomBatch", trainer.RandomBatch.ToString());
            XmlStorage.AddChild(el, "PrintSteps", trainer.PrintSteps);
            XmlStorage.AddChild(el, "LastTrainLoss", trainer.LastTrainLoss);
            XmlStorage.AddChild(el, "LastTrainAccuracy", trainer.LastTrainAccuracy);
            XmlStorage.AddChild(el, "LastTestLoss", trainer.LastTestLoss);
            XmlStorage.AddChild(el, "LastTestAccuracy", trainer.LastTestAccuracy);
            XmlStorage.AddObjectChild(el, "PreProcessor", trainer.PreProcessor);
            XmlStorage.AddObjectChild(el, "LabelCodec", trainer.LabelCodec);
            XmlStorage.AddObjectChild(el, "Normalizer", trainer.Normalizer);
            XmlStorage.AddObjectChild(el, "Model", trainer.Model);

            return el;
        }

        public override object Load(XmlNode node)
        {
            if (node == null) return null;

            var model = XmlStorage.GetObjectValue<IModel>(node, "Model");
            var batchSize = XmlStorage.GetIntValue(node, "BatchSize");
            var epoch = XmlStorage.GetIntValue(node, "Epoch");
            var randomBatch = bool.Parse(XmlStorage.GetStringValue(node, "RandomBatch"));
            var result = (Trainer)Activator.CreateInstance(Type, model, batchSize, epoch, randomBatch);

            result.Mission = XmlStorage.GetStringValue(node, "Mission");
            result.PrintSteps = XmlStorage.GetIntValue(node, "PrintSteps");
            result.LastTrainLoss = XmlStorage.GetDoubleValue(node, "LastTrainLoss");
            result.LastTrainAccuracy = XmlStorage.GetDoubleValue(node, "LastTrainAccuracy");
            result.LastTestLoss = XmlStorage.GetDoubleValue(node, "LastTestLoss");
            result.LastTestAccuracy = XmlStorage.GetDoubleValue(node, "LastTestAccuracy");
            result.PreProcessor = XmlStorage.GetObjectValue<IPreProcessor>(node, "PreProcessor");
            result.LabelCodec = XmlStorage.GetObjectValue<DiscreteCodec>(node, "LabelCodec");
            result.Normalizer = XmlStorage.GetObjectValue<INormalizer>(node, "Normalizer");

            return result;
        }
    }
}
