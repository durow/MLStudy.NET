﻿using MLStudy;
using MLView.MVVM;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLView.Models
{
    public class LinearConfig : NotificationObject
    {
        private double learningRate = 0.0001;

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

        private double regularizationWeight = 0.01;

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

        public void SetToModel(LinearRegression lr)
        {
            lr.LearningRate = LearningRate;
            lr.Regularization = WeightDecay.Get(Regularization);
            lr.Regularization.Strength = RegularizationWeight;
        }

        private WeightDecayTypes GetRegType(string regString)
        {
            if (regString == "L1")
                return WeightDecayTypes.L1;
            if (regString == "L2")
                return WeightDecayTypes.L2;
            return WeightDecayTypes.None;
        }
    }
}
