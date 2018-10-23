using MLStudy;
using MLView.MVVM;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLView.Models
{
    public class TrainerConfig : NotificationObject
    {
        private int maxSteps = 0;

        public int MaxSteps
        {
            get { return maxSteps; }
            set
            {
                if (maxSteps != value)
                {
                    maxSteps = value;
                    RaisePropertyChanged("MaxSteps");
                }
            }
        }

        private int batchSize = 0;

        public int BatchSize
        {
            get { return batchSize; }
            set
            {
                if (batchSize != value)
                {
                    batchSize = value;
                    RaisePropertyChanged("BatchSize");
                }
            }
        }

        private double errorLimit = 0;

        public double ErrorLimit
        {
            get { return errorLimit; }
            set
            {
                if (errorLimit != value)
                {
                    errorLimit = value;
                    RaisePropertyChanged("ErrorLimit");
                }
            }
        }

        private int notifySteps = 100;

        public int NotifySteps
        {
            get { return notifySteps; }
            set
            {
                if (notifySteps != value)
                {
                    notifySteps = value;
                    RaisePropertyChanged("NotifySteps");
                }
            }
        }

        private int stepWait = 0;

        public int StepWait
        {
            get { return stepWait; }
            set
            {
                if (stepWait != value)
                {
                    stepWait = value;
                    RaisePropertyChanged("StepWait");
                }
            }
        }


        public void SetToTrainer(Trainer trainer)
        {
            trainer.BatchSize = BatchSize;
            trainer.ErrorLimit = ErrorLimit;
            trainer.MaxStep = MaxSteps;
            trainer.NotifySteps = NotifySteps;
            trainer.StepWait = StepWait;
        }
    }
}
