# MLStudy.NET

This is a project for machine learning study, especially for deep learning.

I love C# so I write it by pure C#, no 3rd part dependency. It is easy to rewrite to Python or other language.

# Description

```C#

//create a fully connected neural network
var nn = new NeuralNetwork()
            .AddFullLayer(100) //fully connected layer with 100 units
            .AddReLU()  //ReLU activation
            .AddFullLayer(10)
            .AddSoftmax() //softmax output
            .UseAdam()  //use Adam optimizer
            .UseCrossEntropyLoss(); //use cross entropy loss function

//create a Trainer to train the model
var trainer = new Trainer(nn, batchSize = 64, epoch = 10, randomBatch = true)
            {
                LabelCodec = codec,  //set label codec
                Normalizer = norm,   //set normalizer
            };

trainer.StartTrain(trainX, trainY, testX, testY);

//get the machine after train
trainer.GetClassificationMachine();

//you can also save the training result to a file
Storage.Save(trainer, "filename");

//and load it from file
var trainer = Storage.Load<Trainer>("filename");

//you can save and load models also
Storage.Save(nn, "filename");
var model = Storage.Load<NeuralNetwork>("filename");

```

There're a lot of objects can be stored.

The storage file is xml format:

```XML

<?xml version="1.0" encoding="utf-8"?>
<Trainer>
  <Mission>MNIST</Mission>
  <BatchSize>64</BatchSize>
  <Epoch>10</Epoch>
  <RandomBatch>False</RandomBatch>
  <PrintSteps>10</PrintSteps>
  <LastTrainLoss>0</LastTrainLoss>
  <LastTrainAccuracy>0</LastTrainAccuracy>
  <LastTestLoss>0</LastTestLoss>
  <LastTestAccuracy>0</LastTestAccuracy>
  <PreProcessor />
  <LabelCodec>
    <OneHotCodec>a,b,c</OneHotCodec>
  </LabelCodec>
  <Normalizer />
  <Model>
    <NeuralNetwork>
      <LossFunction>
        <CrossEntropy />
      </LossFunction>
      <Optimizer>
        <Adam>
          <Alpha>0.001</Alpha>
          <Beta1>0.9</Beta1>
          <Beta2>0.999</Beta2>
        </Adam>
      </Optimizer>
      <Regularizer />
      <Layers>
        <FullLayer>
          <UnitCount>10</UnitCount>
          <Weights />
          <Bias />
        </FullLayer>
        <Sigmoid />
        <FullLayer>
          <UnitCount>6</UnitCount>
          <Weights />
          <Bias />
        </FullLayer>
        <Sigmoid />
        <FullLayer>
          <UnitCount>3</UnitCount>
          <Weights />
          <Bias />
        </FullLayer>
        <Softmax />
      </Layers>
    </NeuralNetwork>
  </Model>
</Trainer>

```

of cause you can write a xml file directly and use Storage.Load<Trainer>("filename") to load it, as long as you like， but personally i don't like this way ;)

# Demo

https://github.com/durow/MLStudy.NET/tree/master/MNISTDemo
