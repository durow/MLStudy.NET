using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy.DataReader.Csv
{
    public class CsvLineWriter
    {
        private char separator;

        public CsvLineWriter(char separator = ',')
        {
            this.separator = separator;
        }

        public string WriteLine(IEnumerable<string> fields)
        {
            var list = new List<string>();
            foreach (var field in fields)
            {
                list.Add(StandardField(field));
            }
            return string.Join(separator.ToString(), list);
        }

        private string StandardField(string field)
        {
            var quotes = false;

            if (field.Contains(separator.ToString()))
            {
                quotes = true;
            }

            if(field.Contains("\""))
            {
                field = field.Replace("\"", "\"\"");
                quotes = true;
            }

            return quotes ? $"\"{field}\"" : field;
        }
    }
}
