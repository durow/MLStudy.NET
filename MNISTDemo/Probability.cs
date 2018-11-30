using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MNISTDemo
{
    public class Probability : BindingBase
    {
        public string Category { get; set; }

        private string codec;

        public string Codec
        {
            get { return codec; }
            set
            {
                if (codec != value)
                {
                    codec = value;
                    RaisePropertyChanged("Codec");
                }
            }
        }

        private string p;

        public string P
        {
            get { return p; }
            set
            {
                if(p != value)
                {
                    p = value;
                    RaisePropertyChanged("P");
                }
            }
        }

    }
}
