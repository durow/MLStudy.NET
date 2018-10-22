using MLView.MVVM;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLView.Models
{
    public class LinearRegressionConfig : NotificationObject
    {
        private double learningRate;

        public double LearningRate
        {
            get { return learningRate; }
            set
            {
                if(LearningRate != value)
                {
                    learningRate = value;
                    RaisePropertyChanged("LearningRate");
                }
            }
        }

        private bool isRegularization;

        public bool IsRegularization
        {
            get { return isRegularization; }
            set
            {
                if (isRegularization != value)
                {
                    isRegularization = value;
                    RaisePropertyChanged("IsRegularization");
                }
            }
        }

        private string regularization;

        public string Regularization
        {
            get { return regularization; }
            set
            {
                if (regularization != value)
                {
                    regularization = value;
                    RaisePropertyChanged("Regularization");
                }
            }
        }

        private double regularizationWeight;

        public double RegularizationWeight
        {
            get { return regularizationWeight; }
            set
            {
                if (regularizationWeight != value)
                {
                    regularizationWeight = value;
                    RaisePropertyChanged("RegularizationWeight");
                }
            }
        }

    }
}
