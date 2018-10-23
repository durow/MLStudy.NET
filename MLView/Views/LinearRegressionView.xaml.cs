using MLStudy;
using MLView.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
//using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
//using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace MLView.Views
{
    /// <summary>
    /// LinearRegressionView.xaml 的交互逻辑
    /// </summary>
    public partial class LinearRegressionView : UserControl
    {
        LinearRegression lr;
        Trainer trainer;
        DataEmulator emu = new DataEmulator();
        Matrix trainX, testX;
        Vector trainY, testY;

        public LinearRegressionView()
        {
            InitializeComponent();

            TrainerConfig.Config =new TrainerConfig();
            DataConfig.Config = new DataConfig();
            LinearRegConfig.Config = new LinearRegressionConfig();
            lr = new LinearRegression();
            trainer = new Trainer(lr, Loss.MeanSquareError);
            TrainerControl.Trainer = trainer;

            trainer.BeforeStart += Trainer_BeforeStart;
            trainer.Started += Trainer_Started;
            trainer.Stopped += Trainer_Stopped;
            trainer.Notify += Trainer_Notify;
            trainer.Paused += Trainer_Paused;
            trainer.Continued += Trainer_Continued;
        }

        private void Trainer_BeforeStart(object sender, EventArgs e)
        {
            var trainX = emu.RandomMatrix(DataConfig.Config.TrainSize, 1);
            var trainY = (trainX * 3 + 5).ToVector();
            
            if(DataConfig.Config.IsNoise)
            {
                var noise = emu.RandomVectorGaussian(trainY.Length, DataConfig.Config.NoiseMean, DataConfig.Config.NoiseVar);
                trainY += noise;
            }

            lr.SetWeights(0);
            lr.SetBias(1);
            
        }

        private void Trainer_Continued(object sender, EventArgs e)
        {
            throw new NotImplementedException();
        }

        private void Trainer_Paused(object sender, EventArgs e)
        {
            throw new NotImplementedException();
        }

        private void Trainer_Notify(object sender, NotifyEventArgs e)
        {
            throw new NotImplementedException();
        }

        private void Trainer_Stopped(object sender, EventArgs e)
        {
            throw new NotImplementedException();
        }

        private void Trainer_Started(object sender, EventArgs e)
        {
            throw new NotImplementedException();
        }

        private void ConfigModel(LinearRegressionConfig config)
        {
            lr.SetWeights(0);
            lr.SetBias(1);
            lr.LearningRate = config.LearningRate;
            lr.Regularization = (LinearRegularization)Enum.Parse(typeof(LinearRegularization), config.Regularization);
            lr.RegularizationWeight = config.RegularizationWeight;
        }

        private void ConfigTrainer(TrainerConfig config)
        {
            trainer.BatchSize = config.BatchSize;
            trainer.ErrorLimit = config.ErrorLimit;
            trainer.MaxStep = config.MaxSteps;
            trainer.NotifySteps = config.NotifySteps;
        }

        private void SetData(DataConfig config)
        { }
    }
}
