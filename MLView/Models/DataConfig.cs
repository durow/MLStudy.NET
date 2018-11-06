using MLStudy;
using MLView.MVVM;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLView.Models
{
    public class DataConfig : NotificationObject
    {
        private DataEmulator emu = new DataEmulator();

        private double min = 0;

        public double Min
        {
            get { return min; }
            set
            {
                if(min != value)
                {
                    min = value;
                    RaisePropertyChanged("Min");
                }
            }
        }

        private double max = 100;

        public double Max
        {
            get { return max; }
            set
            {
                if (max != value)
                {
                    max = value;
                    RaisePropertyChanged("Max");
                }
            }
        }

        private int trainSize = 20;

        public int TrainSize
        {
            get { return trainSize; }
            set
            {
                if (trainSize != value)
                {
                    trainSize = value;
                    RaisePropertyChanged("TrainSize");
                }
            }
        }

        private int testSize = 20;

        public int TestSize
        {
            get { return testSize; }
            set
            {
                if (testSize != value)
                {
                    testSize = value;
                    RaisePropertyChanged("TestSize");
                }
            }
        }

        private bool isNoise = false;

        public bool IsNoise
        {
            get { return isNoise; }
            set
            {
                if (isNoise != value)
                {
                    isNoise = value;
                    RaisePropertyChanged("IsNoise");
                }
            }
        }

        private double noiseMean = 0;

        public double NoiseMean
        {
            get { return noiseMean; }
            set
            {
                if (noiseMean != value)
                {
                    noiseMean = value;
                    RaisePropertyChanged("NoiseMean");
                }
            }
        }

        private double noiseVar = 1;

        public double NoiseVar
        {
            get { return noiseVar; }
            set
            {
                if (noiseVar != value)
                {
                    noiseVar = value;
                    RaisePropertyChanged("NoiseVar");
                }
            }
        }

        public (Matrix,Vector,Matrix,Vector) GetRegressionData(int features, Func<Matrix,Vector> mapping)
        {
            var trainX = emu.RandomMatrix(trainSize, features, Min, Max);
            var trainY = mapping(trainX);

            if(IsNoise)
            {
                var noise = emu.RandomVectorGaussian(trainY.Length, NoiseMean, NoiseVar);
                trainY += noise;
            }

            Matrix testX = null;
            Vector testY = null;

            if (TestSize > 0)
            {
                testX = emu.RandomMatrix(TestSize, features, Min, Max);
                testY = mapping(testX);
            }

            return (trainX, trainY, testX, testY);
        }

        public (Matrix, Vector, Matrix, Vector) GetLogisticData(int features, Func<Matrix,Vector> distance)
        {
            var trainX = emu.RandomMatrix(trainSize, features, Min, Max);
            var trainDistance = distance(trainX);
            if (IsNoise)
            {
                var noise = emu.RandomVectorGaussian(trainDistance.Length, NoiseMean, NoiseVar);
                trainDistance += noise;
            }

            var trainY = trainDistance.ApplyFunction(Functions.IndicatorFunction);
            
            Matrix testX = null;
            Vector testY = null;

            if (TestSize > 0)
            {
                testX = emu.RandomMatrix(TestSize, features, Min, Max);
                var testDistance = distance(testX);
                testY = testDistance.ApplyFunction(Functions.IndicatorFunction);
            }

            return (trainX, trainY, testX, testY);
        }

        public (Matrix, Vector, Matrix, Vector) GetClassificationData(int features, Func<Matrix, Vector> classify)
        {
            var trainX = emu.RandomMatrix(trainSize, features, Min, Max);
            
            if (IsNoise)
            {
                var noise = emu.RandomMatrixGaussian(trainX.Rows, trainX.Columns, NoiseMean, NoiseVar);
                trainX += noise;
            }
            var trainY = classify(trainX);

            Matrix testX = null;
            Vector testY = null;

            if (TestSize > 0)
            {
                testX = emu.RandomMatrix(TestSize, features, Min, Max);
                testY = classify(testX);
            }

            return (trainX, trainY, testX, testY);
        }
    }
}
