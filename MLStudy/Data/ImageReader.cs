/*
 * Description:Read image files for DNN/CNN, based on SixLabors.ImageSharp
 * Author:Yunxiao An
 * Date:2018.11.05
 */

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace MLStudy.Data
{
    public class ImageReader
    {
        public static Matrix ReadFile8BitToMatrix(string filename)
        {
            var img = Image.Load<Alpha8>(filename);
            var result = new Matrix(img.Height, img.Width);
            for (int i = 0; i < result.Rows; i++)
            {
                for (int j = 0; j < result.Columns; j++)
                {
                    result[i, j] = img[j, i].PackedValue;
                }
            }
            return result;
        }

        public static Matrix ReadDir8BitToMatrix(string dir, params string[] extensions)
        {
            extensions = GetExtensions(extensions);
            var imgs = GetImagesFromDir<Alpha8>(dir, extensions);
            var width = imgs[0].Width;
            var height = imgs[0].Height;
            var length = width * height;
            var result = new Matrix(imgs.Count, length);
            for (int i = 0; i < result.Rows; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    for (int k = 0; k < width; k++)
                    {
                        result[i, j * width + k] = imgs[i][k, j].PackedValue;
                    }
                }
            }
            return result;
        }

        public static Tensor3 ReadDir8BitToTensor3(string dir, params string[] extensions)
        {
            extensions = GetExtensions(extensions);
            var imgs = GetImagesFromDir<Alpha8>(dir, extensions);
            var width = imgs[0].Width;
            var height = imgs[0].Height;
            var length = width * height;

            var result = new Tensor3(imgs.Count, height, width);
            for (int i = 0; i < result.Rows; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    for (int k = 0; k < width; k++)
                    {
                        result[i, j, k] = imgs[i][k, j].PackedValue;
                    }
                }
            }
            return result;
        }

        public static Tensor3 ReadFileRgb24ToTensor3(string filename)
        {
            var img = Image.Load<Rgb24>(filename);
            var result = new Tensor3(3, img.Height, img.Width);
            for (int i = 0; i < result.Rows; i++)
            {
                for (int j = 0; j < result.Columns; j++)
                {
                    var pixel = img[j, i];
                    result[0, i, j] = pixel.R;
                    result[1, i, j] = pixel.G;
                    result[2, i, j] = pixel.B;
                }
            }
            return result;
        }

        public static Tensor4 ReadDirRgb24ToTensor4(string dir, params string[] extensions)
        {
            extensions = GetExtensions(extensions);
            var imgs = GetImagesFromDir<Rgb24>(dir, extensions);

            var result = new Tensor4(imgs.Count, 3, imgs[0].Height, imgs[1].Width);
            for (int i = 0; i < imgs.Count; i++)
            {
                for (int j = 0; j < result.D3; j++)
                {
                    for (int k = 0; k < result.D4; k++)
                    {
                        result[i, 0, j, k] = imgs[i][k, j].R;
                        result[i, 1, j, k] = imgs[i][k, j].G;
                        result[i, 2, j, k] = imgs[i][k, j].B;
                    }

                }
            }
            return result;
        }

        private static string[] GetExtensions(params string[] extensions)
        {
            if (extensions.Length == 0)
                return extensions = new string[] { ".jpg", ".png", ".bmp", "jpeg" };
            else
                return extensions;
        }

        public static List<Image<T>> GetImagesFromDir<T>(string dir, params string[] extensions)
            where T:struct,IPixel<T>
        {
            return Directory.GetFiles(dir)
                .Where(f =>
                {
                    var ext = Path.GetExtension(f).ToLower();
                    return extensions.Contains(ext);
                })
                .Select(f => Image.Load<T>(f))
                .ToList();
        }
    }
}
