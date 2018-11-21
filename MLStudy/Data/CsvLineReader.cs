/*
 * Read csv line as List<string> using separator
 * Author:durow
 * Date:2018.03
 */

using System.Collections.Generic;
using System.Text;

namespace MLStudy.Data
{
    public class CsvLineReader
    {
        private StringBuilder field;
        private int pointer;
        private ReadState state;
        private List<string> result;
        private readonly char separator;

        public CsvLineReader(char separator = ',')
        {
            this.separator = separator;
        }
         
        public List<string> ReadLine(string line)
        {
            Reset();

            while (true)
            {
                switch (state)
                {
                    case ReadState.Find:
                        FindHandler(line);
                        break;
                    case ReadState.Read:
                        ReadHandler(line);
                        break;
                    case ReadState.InField:
                        InFieldHandler(line);
                        break;
                    default:
                        break;
                }

                if (pointer == line.Length)
                {
                    AddFiled();
                    break;
                }
            }
            return result;
        }

        private void FindHandler(string line)
        {
            var c = line[pointer];

            if(c == '"')
            {
                state = ReadState.InField;
            }
            else if(c == separator)
            {
                AddFiled();
            }
            else
            {
                field.Append(c);
                state = ReadState.Read;
            }

            pointer++;
        }

        private void ReadHandler(string line)
        {
            var c = line[pointer];

            if(c == separator)
            {
                AddFiled();
                state = ReadState.Find;
            }
            else
            {
                field.Append(c);
            }

            pointer++;
        }

        private void InFieldHandler(string line)
        {
            var c = line[pointer];
            
            if(c == '"')
            {
                if (pointer == line.Length - 1)
                {
                    pointer++;
                    return;
                }

                var next = line[pointer + 1];
                if (next == '"')
                {
                    field.Append(c);
                    pointer++;
                }
                else if (next == separator)
                {
                    AddFiled();
                    state = ReadState.Find;
                    pointer++;
                }
                else
                    field.Append(c);
            }
            else
            {
                field.Append(c);
            }

            pointer++;
        }

        private void AddFiled()
        {
            result.Add(field.ToString());
            field.Clear();
        }

        private void Reset()
        {
            pointer = 0;
            state = ReadState.Find;
            result = new List<string>();
            field = new StringBuilder();
        }
    }

    enum ReadState
    {
        Find,
        Read,
        InField,
    }
}
