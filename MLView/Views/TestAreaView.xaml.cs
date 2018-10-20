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

namespace MLView
{
    /// <summary>
    /// TestAreaView.xaml 的交互逻辑
    /// </summary>
    public partial class TestAreaView : UserControl
    {
        public TestAreaView()
        {
            InitializeComponent();
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            var count = 50;
            var emu = new DataEmulator();
            var X = emu.RandomVector(count).ApplyFunction(a => (int)(a * 100));
            var y = X.ApplyFunction(F);
            var noise = emu.RandomVectorGaussian(count) * 10;
            var noiseY = y + noise;

            //var lr = new LinearRegression();
            //lr.Step(X, y);
        }

        private void TextOut(string text)
        {
            OutText.Text += text + "\n";
        }

        private double F(double x)
        {
            return 3 * x + 5;
        }
    }
}
