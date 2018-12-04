using System;
using System.Collections.Generic;
using System.Data;
using System.Text;

namespace MLStudy.Data
{
    public class DataFormat
    {
        public static TensorOld DataTableToTensor(DataTable table, params int[] columns)
        {
            var result = new TensorOld(table.Rows.Count, columns.Length);
            DataTableToTensor(table, result, columns);
            return result;
        }

        public static TensorOld DataTableToTensor(DataTable table, params string[] columns)
        {
            var result = new TensorOld(table.Rows.Count, columns.Length);
            DataTableToTensor(table, result, columns);
            return result;
        }

        public static void DataTableToTensor(DataTable table, TensorOld result, params int[] columns)
        {
            for (int i = 0; i < table.Rows.Count; i++)
            {
                for (int j = 0; j < columns.Length; j++)
                {
                    result[i, j] = double.Parse(table.Rows[i][columns[j]].ToString());
                }
            }
        }

        public static void DataTableToTensor(DataTable table, TensorOld result, params string[] columns)
        {
            for (int i = 0; i < table.Rows.Count; i++)
            {
                for (int j = 0; j < columns.Length; j++)
                {
                    result[i, j] = double.Parse(table.Rows[i][columns[j]].ToString());
                }
            }
        }

        public static IEnumerable<string> DataTableToList(DataTable table, int columnIndex)
        {
            for (int i = 0; i < table.Rows.Count; i++)
            {
                yield return table.Rows[i][columnIndex].ToString();
            }
        }

        public static IEnumerable<string> DataTableToList(DataTable table, string columnName)
        {
            for (int i = 0; i < table.Rows.Count; i++)
            {
                yield return table.Rows[i][columnName].ToString();
            }
        }
    }
}
