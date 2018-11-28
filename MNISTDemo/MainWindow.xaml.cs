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
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace MNISTDemo
{
    /// <summary>
    /// MainWindow.xaml 的交互逻辑
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            WritingBoard.Strokes.Clear();
        }

        private void Button_Click_1(object sender, RoutedEventArgs e)
        {
            RenderTargetBitmap bmp = new RenderTargetBitmap((int)WritingBoard.ActualWidth, (int)WritingBoard.ActualHeight, 0, 0, PixelFormats.Default);
            bmp.Render(WritingBoard);

            //
            var con = new FormatConvertedBitmap();
            con.BeginInit();
            con.Source = bmp;
            con.DestinationFormat = PixelFormats.Gray8;
            con.EndInit();
            Test.Source = con;

            var trans = new TransformedBitmap();
            trans.BeginInit();
            trans.Source = con;
            trans.Transform = new ScaleTransform(0.1, 0.1);
            trans.EndInit();

            var buff = BitmapSourceToArray(trans);
            for (int i = 0; i < buff.Length; i++)
            {
                buff[i] = (byte)(255 - buff[i]);
            }
            
        }

        private byte[] BitmapSourceToArray(BitmapSource bitmapSource)
        {
            // Stride = (width) x (bytes per pixel)
            int stride = (int)bitmapSource.PixelWidth * (bitmapSource.Format.BitsPerPixel / 8);
            byte[] pixels = new byte[(int)bitmapSource.PixelHeight * stride];

            bitmapSource.CopyPixels(pixels, stride, 0);

            return pixels;
        }

        private BitmapSource BitmapSourceFromArray(byte[] pixels, int width, int height)
        {
            WriteableBitmap bitmap = new WriteableBitmap(width, height, 96, 96, PixelFormats.Bgra32, null);

            bitmap.WritePixels(new Int32Rect(0, 0, width, height), pixels, width * (bitmap.Format.BitsPerPixel / 8), 0);

            return bitmap;
        }
    }
}
