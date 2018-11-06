using MLStudy;
using MLStudy.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace MLView
{
    /// <summary>
    /// TestAreaView.xaml 的交互逻辑
    /// </summary>
    public partial class TestAreaView : UserControl
    {


        public int Range
        {
            get { return (int)GetValue(RangeProperty); }
            set { SetValue(RangeProperty, value); }
        }

        // Using a DependencyProperty as the backing store for Range.  This enables animation, styling, binding, etc...
        public static readonly DependencyProperty RangeProperty =
            DependencyProperty.Register("Range", typeof(int), typeof(TestAreaView), new PropertyMetadata(100));



        public double LearningRate
        {
            get { return (double)GetValue(LearningRateProperty); }
            set { SetValue(LearningRateProperty, value); }
        }

        // Using a DependencyProperty as the backing store for LearningRate.  This enables animation, styling, binding, etc...
        public static readonly DependencyProperty LearningRateProperty =
            DependencyProperty.Register("LearningRate", typeof(double), typeof(TestAreaView), new PropertyMetadata(0.0001));



        public int TrainCount
        {
            get { return (int)GetValue(TrainCountProperty); }
            set { SetValue(TrainCountProperty, value); }
        }

        // Using a DependencyProperty as the backing store for TrainCount.  This enables animation, styling, binding, etc...
        public static readonly DependencyProperty TrainCountProperty =
            DependencyProperty.Register("TrainCount", typeof(int), typeof(TestAreaView), new PropertyMetadata(1000));



        public double Weight
        {
            get { return (double)GetValue(WeightProperty); }
            set { SetValue(WeightProperty, value); }
        }

        // Using a DependencyProperty as the backing store for Weight.  This enables animation, styling, binding, etc...
        public static readonly DependencyProperty WeightProperty =
            DependencyProperty.Register("Weight", typeof(double), typeof(TestAreaView), new PropertyMetadata(3d));



        public int SampleCount
        {
            get { return (int)GetValue(SampleCountProperty); }
            set { SetValue(SampleCountProperty, value); }
        }

        // Using a DependencyProperty as the backing store for SampleCount.  This enables animation, styling, binding, etc...
        public static readonly DependencyProperty SampleCountProperty =
            DependencyProperty.Register("SampleCount", typeof(int), typeof(TestAreaView), new PropertyMetadata(20));



        public double Bias
        {
            get { return (double)GetValue(BiasProperty); }
            set { SetValue(BiasProperty, value); }
        }

        // Using a DependencyProperty as the backing store for Bias.  This enables animation, styling, binding, etc...
        public static readonly DependencyProperty BiasProperty =
            DependencyProperty.Register("Bias", typeof(double), typeof(TestAreaView), new PropertyMetadata(5d));



        public int ReportStep
        {
            get { return (int)GetValue(ReportStepProperty); }
            set { SetValue(ReportStepProperty, value); }
        }

        // Using a DependencyProperty as the backing store for ReportStep.  This enables animation, styling, binding, etc...
        public static readonly DependencyProperty ReportStepProperty =
            DependencyProperty.Register("ReportStep", typeof(int), typeof(TestAreaView), new PropertyMetadata(100));



        public int TestSize
        {
            get { return (int)GetValue(TestSizeProperty); }
            set { SetValue(TestSizeProperty, value); }
        }

        // Using a DependencyProperty as the backing store for TestSize.  This enables animation, styling, binding, etc...
        public static readonly DependencyProperty TestSizeProperty =
            DependencyProperty.Register("TestSize", typeof(int), typeof(TestAreaView), new PropertyMetadata(10));

        LinearRegression lr;
        Trainer trainer;

        public TestAreaView()
        {
            InitializeComponent();
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            var file = @"D:\AYX\Data\train-labels.idx1-ubyte";
            var matrix = MINISTReader.ReadLabels(file);
        }

        private void Trainer_Notify(object sender, NotifyEventArgs e)
        {
            var yHat = lr.Predict(e.X);
            var error = LossFunctions.MeanSquareError(yHat, e.Y);

            TextOutCross($"Step:{e.Step}, error:{error}!");
        }

        private void Trainer_Stopped(object sender, EventArgs e)
        {
            TextOutCross("Stopped! "+trainer.State);
        }

        private void Trainer_Started(object sender, EventArgs e)
        {
            TextOutCross("Started!");
        }

        private void TextOutCross(string text)
        {
            Dispatcher.Invoke(() =>
            {
                TextOut(text);
            });
        }

        private void TextOut(string text)
        {
            OutText.Text += text + "\n";
        }

        private double F(double x)
        {
            return Weight * x + Bias;
        }

        private void ShowTrainInfo(LinearRegression lr, Matrix X, MLStudy.Vector y, Matrix testX, MLStudy.Vector testY)
        {
            var yHat = lr.Predict(X);
            var testYHat = lr.Predict(testX);
            var trainError = LossFunctions.MeanSquareError(yHat, y);
            var testError = LossFunctions.MeanSquareError(testYHat, testY);
            TextOutCross($"step:,weight:{lr.Weights}, bias:{lr.Bias}, trainError:{trainError}, testError:{testError}");
        }

        private void StartButton_Click(object sender, RoutedEventArgs e)
        {

        }

        private void StopButton_Click(object sender, RoutedEventArgs e)
        {
            trainer?.Stop();
        }

        private void PauseButton_Click(object sender, RoutedEventArgs e)
        {
            trainer?.Pause();
        }

        private void ContinueButton_Click(object sender, RoutedEventArgs e)
        {
            Task.Factory.StartNew(() =>
            {
                trainer?.Continue();
            });
        }
    }
}
