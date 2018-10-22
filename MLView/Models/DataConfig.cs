using MLView.MVVM;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLView.Models
{
    public class DataConfig : NotificationObject
    {
        private double min = 0;

        public double Min
        {
            get { return min; }
            set
            {
                if(min != value)
                {
                    min = value;
                    RaisePropertyChanged("Min");
                }
            }
        }

        private double max = 100;

        public double Max
        {
            get { return max; }
            set
            {
                if (max != value)
                {
                    max = value;
                    RaisePropertyChanged("Max");
                }
            }
        }

        private int trainSize = 100;

        public int TrainSize
        {
            get { return trainSize; }
            set
            {
                if (max != value)
                {
                    max = value;
                    RaisePropertyChanged("Max");
                }
            }
        }

        private int testSize = 20;

        public int TestSize
        {
            get { return testSize; }
            set
            {
                if (testSize != value)
                {
                    testSize = value;
                    RaisePropertyChanged("TestSize");
                }
            }
        }

    }
}
