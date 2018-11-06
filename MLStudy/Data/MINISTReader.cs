/*
 * FILE FORMATS FOR THE MNIST DATABASE
 *   The data is stored in a very simple file format designed for storing vectors and multidimensional matrices. 
 *   General info on this format is given at the end of this page, but you don't need to read that to use the data files.
 *   All the integers in the files are stored in the MSB first (high endian) format used by most non-Intel processors. 
 *   Users of Intel processors and other low-endian machines must flip the bytes of the header.
 *
 *    There are 4 files:
 *
 *   train-images-idx3-ubyte: training set images 
 *   train-labels-idx1-ubyte: training set labels 
 *   t10k-images-idx3-ubyte:  test set images 
 *   t10k-labels-idx1-ubyte:  test set labels
 *
 *   The training set contains 60000 examples, and the test set 10000 examples.
 *
 *   The first 5000 examples of the test set are taken from the original NIST training set.
 *   The last 5000 are taken from the original NIST test set. The first 5000 are cleaner and easier than the last 5000.
 */

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace MLStudy.Data
{
    public class MINISTReader
    {
        public string TrainImagesFile { get; set; }
        public string TrainLabelsFile { get; set; }
        public string TestImagesFile { get; set; }
        public string TestLabelsFile { get; set; }

        public Matrix ReadTrainImagesToMatrix()
        {
            throw new NotImplementedException();
        }

        public Vector ReadTrainLabels()
        {
            throw new NotImplementedException();
        }

        public Tensor3 ReadTrainImagesToTensor3()
        {
            throw new NotImplementedException();
        }

        public Vector ReadTestLabels()
        {
            throw new NotImplementedException();
        }

        /*
         * TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
         *  [offset] [type]          [value]          [description] 
         *   0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
         *   0004     32 bit integer  60000            number of items 
         *   0008     unsigned byte   ??               label 
         *   0009     unsigned byte   ??               label 
         *   ........ 
         *   xxxx     unsigned byte   ??               label
         *   The labels values are 0 to 9.
         */

        public static Vector ReadLabels(string filename)
        {
            using (var fs = new FileStream(filename, FileMode.Open))
            {
                using (var br = new BinaryReader(fs))
                {
                    br.ReadInt32(); //skip the 32bit magic number
                    var count = ReadInt32BigEndian(br); //read the number of labels
                    var result = new Vector(count);

                    for (int i = 0; i < count; i++)
                    {
                        result[i] = br.ReadByte();
                    }
                    return result;
                }
            }
        }

        public static int ReadInt32BigEndian(BinaryReader br)
        {
            var buff = new byte[4];
            buff[3] = br.ReadByte();
            buff[2] = br.ReadByte();
            buff[1] = br.ReadByte();
            buff[0] = br.ReadByte();
            return BitConverter.ToInt32(buff, 0);
        }

        /*
         * TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
         *   [offset] [type]          [value]          [description] 
         *   0000     32 bit integer  0x00000803(2051) magic number 
         *   0004     32 bit integer  60000            number of images 
         *   0008     32 bit integer  28               number of rows 
         *   0012     32 bit integer  28               number of columns 
         *   0016     unsigned byte   ??               pixel 
         *   0017     unsigned byte   ??               pixel 
         *   ........ 
         *   xxxx     unsigned byte   ??               pixel
         *   Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
         */

        public static Matrix ReadImagesToMatrix(string filename)
        {
            throw new NotImplementedException();
        }

        public static Tensor3 ReadImagesToTensor3(string filename)
        {
            throw new NotImplementedException();
        }

        public static Tensor4 ReadImagesToTensor4(string filename)
        {
            throw new NotImplementedException();
        }
    }
}
