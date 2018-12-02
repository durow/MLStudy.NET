using MLStudy.Abstraction;
using MLStudy.Storages.Xml;
using System;
using System.Collections.Generic;
using System.Text;
using System.Xml;

namespace MLStudy.Storages
{
    public class XmlMseStorage : XmlStorageBase<MeanSquareError>{}
    public class XmlCrossEntropyStorage : XmlStorageBase<CrossEntropy>{ }
    public class XmlCrossEntropyFromSoftmaxStorage : XmlStorageBase<CrossEntropyFromSoftmax>{ }
}
