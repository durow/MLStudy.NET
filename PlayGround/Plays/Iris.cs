using System;
using System.Collections.Generic;
using System.Data;
using System.Text;
using MLStudy;
using MLStudy.Data;

namespace PlayGround.Plays
{
    public class Iris : IPlay
    {
        public void Play()
        {
            var iris = AyxCsvReader.Instance.ReadCsvFileDataTable("Data/iris.csv");

            Console.WriteLine(iris.Select("Species"));
        }

        public static void DataTableToTensor(DataTable table, Tensor result, params int[] columns)
        {
        }

        //public static List<string> DataTable2List(DataTable table, int columnIndex)
        //{ }
    }
}
