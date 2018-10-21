using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace MLStudy
{
    public class Trainer
    {
        public readonly IMachine Machine;
        private Matrix X;
        private Vector y;

        public int BatchSize { get; set; }
        public int MaxStep { get; set; }
        public double ErrorLimit { get; set; } = 0;
        public int NotifySteps { get; set; } = 100;
        public int StepWait { get; set; } = 0;
        public TimeSpan MaxTime { get; set; } = TimeSpan.FromDays(365);
        public Func<Vector, Vector, double> ErrorFunction { get; set; }

        public int StepCounter { get; private set; } = 0;
        public TrainerState State { get; private set; }
        public DateTime? StartTime { get; private set; }
        public DateTime? StopTime { get; private set; }
        public TimeSpan TrainedTime
        {
            get
            {
                if (State == TrainerState.Training)
                    return DateTime.Now - StartTime.Value;
                else
                    return new TimeSpan(0);
            }
        }

        public event EventHandler Notify;
        public event EventHandler Started;
        public event EventHandler Stopped;

        public Trainer(IMachine machine, Func<Vector,Vector,double> errorFunction)
        {
            Machine = machine;
            ErrorFunction = errorFunction;
            State = TrainerState.Ready;
        }

        public void Train(Matrix X, Vector y)
        {
            if (State == TrainerState.Training)
                return;

            CheckTraningData(X, y);
            this.X = X;
            this.y = y;

            Task.Factory.StartNew(TrainingThread);
        }

        private void TrainingThread()
        {
            StepCounter = 0;
            State = TrainerState.Training;
            Started?.Invoke(this, null);

            Matrix batchX;
            Vector batchY;
            var error = double.MaxValue;

            while (State == TrainerState.Training)
            {
                var batchStart = StepCounter % X.Rows;
                (batchX, batchY) = GetBatchX(X, y, batchStart, BatchSize);
                Machine.Step(batchX, batchY);


                StepCounter++;
                if (StepCounter % NotifySteps == 0)
                    Notify?.Invoke(this, null);

                if (StepCounter >= MaxStep)
                    State = TrainerState.MaxStepsStopped;

                if (ErrorLimit > 0 && ErrorFunction != null)
                {
                    var yHat = Machine.Predict(batchX);
                    error = ErrorFunction(yHat, batchY);
                    if (error <= ErrorLimit)
                        State = TrainerState.ErrorLimitStopped;
                }
                if (MaxTime >= TrainedTime)
                    State = TrainerState.TimeLimitStopped;
            }

            Stopped?.Invoke(this, null);
        }

        public void Pause()
        {
            if (State != TrainerState.Training)
                return;

            State = TrainerState.Paused;
        }

        public void Continue()
        {
            if (State != TrainerState.Paused)
                return;

            Task.Factory.StartNew(TrainingThread);
        }

        public void Stop()
        {
            if (State != TrainerState.Training)
                return;

            State = TrainerState.UserStopped;
        }

        public double Predict(Vector x)
        {
            return Machine.Predict(x);
        }

        public Vector Predict(Matrix X)
        {
            return Machine.Predict(X);
        }

        public static (Matrix,Vector) GetBatchX(Matrix X, Vector y, int start, int count)
        {
            if (count <= 0 || count >= X.Rows)
                return (X, y);

            var batchX = new double[count, X.Columns];
            var batchY = new double[count];
            for (int i = 0; i < count; i++)
            {
                var row = (i + start) % X.Rows;
                for (int j = 0; j < X.Columns; j++)
                {
                    batchX[i, j] = X[row, j];
                }
                batchY[i] = y[row];
            }
            return (new Matrix(batchX), new Vector());
        }

        private void CheckTraningData(Matrix X, Vector y)
        {
            if (X.Rows != y.Length)
                throw new Exception($"the number of training samples in X must equal to number of labels in y!X.Rows:{X.Rows},y.Length:{y.Length}");
        }
    }

    public enum TrainerState
    {
        Training,
        Ready,
        Paused,
        MaxStepsStopped,
        ErrorLimitStopped,
        TimeLimitStopped,
        UserStopped,
        ErrorStopped,
    }
}
