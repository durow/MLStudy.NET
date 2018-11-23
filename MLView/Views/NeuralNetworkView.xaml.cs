using MLStudy;
using MLView.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace MLView.Views
{
    /// <summary>
    /// NeuralNetworkView.xaml 的交互逻辑
    /// </summary>
    public partial class NeuralNetworkView : UserControl
    {
        NeuralNetwork nn;
        Trainer trainer;
        Matrix trainX, testX;
        Vector trainY, testY;

        DataConfig dataConfig = new DataConfig { TrainSize = 200, TestSize = 50, Min = -50, Max = 50 };
        TrainerConfig trainerConfig = new TrainerConfig();
        LinearConfig lrConfig = new LinearConfig { LearningRate = 0.1 };

        public NeuralNetworkView()
        {
            InitializeComponent();

            TrainerConfig.Config = trainerConfig;
            DataConfig.Config = dataConfig;
            LinearRegConfig.Config = lrConfig;
            nn = new NeuralNetwork().UseLogisticRegressionOutLayer();
            trainer = new Trainer(nn);
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
            nn.InitWeightsBias();
            nn.SetLearningRate(lrConfig.LearningRate);

            Dispatcher.Invoke(() =>
            {
                trainerConfig.SetToTrainer(trainer);
                (trainX, trainY, testX, testY) = dataConfig.GetClassificationData(2, m =>
                {
                    var result = new Vector(m.Rows);
                    for (int i = 0; i < m.Rows; i++)
                    {
                        if (m[i, 0] < 0 && m[i, 1] < 0)
                            result[i] = 1;
                        else if (m[i, 0] > 0 && m[i, 1] > 0)
                            result[i] = 1;
                        else
                            result[i] = 0;
                    }
                    return result;
                });

                trainer.SetTrainData(trainX, trainY);
            });
        }

        private void Trainer_Continued(object sender, EventArgs e)
        {
            TextOutCross($"Continued!");
        }

        private void Trainer_Paused(object sender, NotifyEventArgs e)
        {
            TextOutCross($"Paused!");
            OutputInfo(e);
        }

        private void Trainer_Notify(object sender, NotifyEventArgs e)
        {
            OutputInfo(e);
        }

        private void Trainer_Stopped(object sender, NotifyEventArgs e)
        {
            TextOutCross($"Stopped!{trainer.State}!");
            OutputInfo(e);
        }

        private void Button_Click(object sender, System.Windows.RoutedEventArgs e)
        {
            LogText.Clear();
        }

        private void Trainer_Started(object sender, EventArgs e)
        {
            TextOutCross($"Started!");
        }

        private void TextOutCross(string text)
        {
            Dispatcher.Invoke(() =>
            {
                LogText.Text += text + "\n";
                LogText.ScrollToEnd();
            });
        }

        private void OutputInfo(NotifyEventArgs e)
        {
            //nn.Forward(e.X);
            TextOutCross($"Step:{trainer.StepCounter}, Loss:{nn.GetLastLoss(e.Y)} Error:{nn.GetLastError(e.Y)}");
        }
    }
}
