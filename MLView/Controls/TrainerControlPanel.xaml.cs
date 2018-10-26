using MLStudy;
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

namespace MLView.Controls
{
    /// <summary>
    /// TrainerControlPanel.xaml 的交互逻辑
    /// </summary>
    public partial class TrainerControlPanel : UserControl
    {

        private Trainer trainer;

        public Trainer Trainer
        {
            get { return trainer; }
            set
            {
                trainer = value;
                Trainer.Started += Trainer_Started;
                Trainer.Stopped += Trainer_Stopped;
                Trainer.Paused += Trainer_Paused;
                Trainer.Continued += Trainer_Continued;
            }
        }

        public TrainerControlPanel()
        {
            InitializeComponent();
        }

        private void StartButton_Click(object sender, RoutedEventArgs e)
        {
            if (Trainer == null)
            {
                MessageBox.Show("Need a Trainer!");
                return;
            }

            StartButton.IsEnabled = false;
            ContinueButton.IsEnabled = false;

            Task.Factory.StartNew(() =>
            {
                Trainer.Start();
            });
        }

        private void Trainer_Started(object sender, EventArgs e)
        {
            Dispatcher.Invoke(() =>
            {
                StopButton.IsEnabled = true;
                PauseButton.IsEnabled = true;
            });
        }

        private void StopButton_Click(object sender, RoutedEventArgs e)
        {
            StopButton.IsEnabled = false;
            PauseButton.IsEnabled = false;
            ContinueButton.IsEnabled = false;

            Trainer?.Stop();
        }

        private void Trainer_Stopped(object sender, EventArgs e)
        {
            Dispatcher.Invoke(() =>
            {
                StopButton.IsEnabled = false;
                PauseButton.IsEnabled = false;
                ContinueButton.IsEnabled = false;
                StartButton.IsEnabled = true;
            });
        }

        private void PauseButton_Click(object sender, RoutedEventArgs e)
        {
            PauseButton.IsEnabled = false;
            StartButton.IsEnabled = false;

            Trainer?.Pause();
        }

        private void Trainer_Paused(object sender, EventArgs e)
        {
            Dispatcher.Invoke(() =>
            {
                ContinueButton.IsEnabled = true;
                StopButton.IsEnabled = true;
            });
        }

        private void ContinueButton_Click(object sender, RoutedEventArgs e)
        {
            ContinueButton.IsEnabled = false;
            StartButton.IsEnabled = false;

            Task.Factory.StartNew(() =>
            {
                Trainer?.Continue();
            });
        }

        private void Trainer_Continued(object sender, EventArgs e)
        {
            Dispatcher.Invoke(() =>
            {
                StopButton.IsEnabled = true;
                PauseButton.IsEnabled = true;
            });
        }
    }
}
