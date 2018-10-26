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

namespace MLView.Controls
{
    /// <summary>
    /// LinearRegressionPanel.xaml 的交互逻辑
    /// </summary>
    public partial class LinearRegressionPanel : UserControl
    {


        public double TextBlockWidth
        {
            get { return (double)GetValue(TextBlockWidthProperty); }
            set { SetValue(TextBlockWidthProperty, value); }
        }

        // Using a DependencyProperty as the backing store for TextBlockWidth.  This enables animation, styling, binding, etc...
        public static readonly DependencyProperty TextBlockWidthProperty =
            DependencyProperty.Register("TextBlockWidth", typeof(double), typeof(LinearRegressionPanel), new PropertyMetadata(90d));




        public LinearConfig Config
        {
            get { return (LinearConfig)GetValue(ConfigProperty); }
            set { SetValue(ConfigProperty, value); }
        }

        // Using a DependencyProperty as the backing store for Config.  This enables animation, styling, binding, etc...
        public static readonly DependencyProperty ConfigProperty =
            DependencyProperty.Register("Config", typeof(LinearConfig), typeof(LinearRegressionPanel), new PropertyMetadata(null));



        public LinearRegressionPanel()
        {
            InitializeComponent();
        }
    }
}
