using MLStudy;
using MLView.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
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

        public LinearRegressionView()
        {
            InitializeComponent();

            TrainerConfig.Config =new TrainerConfig();
            DataConfig.Config = new DataConfig();
            LinearRegConfig.Config = new LinearRegressionConfig();
            lr = new LinearRegression();
            trainer = new Trainer(lr, Loss.MeanSquareError);
            TrainerControl.Trainer = trainer;

            trainer.Started += Trainer_Started;
            trainer.Stopped += Trainer_Stopped;
            trainer.Notify += Trainer_Notify;
            trainer.Paused += Trainer_Paused;
            trainer.Continued += Trainer_Continued;
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
    }
}
