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



        public TestAreaView()
        {
            InitializeComponent();
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            var emu = new DataEmulator();
            var X = emu.RandomVector(SampleCount).ApplyFunction(a => (int)(a * Range));
            var y = X.ApplyFunction(F);
            var noise = emu.RandomVectorGaussian(SampleCount) * 10;
            var noiseY = y + noise;
            var testX = emu.RandomVector(TestSize).ApplyFunction(a => (int)(a * Range));
            var testY = testX.ApplyFunction(F);


            OutText.Clear();

            TextOut($"True Weight:{Weight},True Bias:{Bias}");

            var lr = new LinearRegression
            {
                LearningRate = LearningRate,
            };
            lr.InitWeights(0);
            lr.InitBias(0);


            var matrixX = X.ToMatrix(true);
            var matrixTestX = testX.ToMatrix(true);

            ShowTrainInfo(lr, matrixX, y, matrixTestX, testY);

            for (int i = 1; i <= TrainCount; i++)
            {
                lr.Step(matrixX, y);
                if (i % ReportStep == 0)
                {
                    ShowTrainInfo(lr, matrixX, y, matrixTestX, testY);
                }
            }

            if (lr.StepCounter % ReportStep != 0)
            {
                ShowTrainInfo(lr, matrixX, y, matrixTestX, testY);
            }
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
            var trainError = Loss.MeanSquareError(yHat, y);
            var testError = Loss.MeanSquareError(testYHat, testY);
            TextOut($"step:{lr.StepCounter},weight:{lr.Weights}, bias:{lr.Bias}, trainError:{trainError}, testError:{testError}");
        }
    }
}
