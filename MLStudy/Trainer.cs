using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MLStudy
{
    public class Trainer
    {
        public readonly IMachine Machine;
        public Matrix X { get; private set; }
        public Vector y { get; private set; }

        public int BatchSize { get; set; }
        public int MaxStep { get; set; }
        public double ErrorLimit { get; set; } = 0;
        public int NotifySteps { get; set; } = 100;
        public int StepWait { get; set; } = 0;
        public Func<Vector, Vector, double> ErrorFunction { get; set; }

        public int StepCounter { get; private set; } = 0;
        public TrainerState State { get; private set; }

        public event EventHandler<NotifyEventArgs> Notify;
        public event EventHandler Started;
        public event EventHandler Stopped;
        public event EventHandler Paused;
        public event EventHandler Continued;
        public event EventHandler ErrorOut;

        public Trainer(IMachine machine, Func<Vector,Vector,double> errorFunction)
        {
            Machine = machine;
            ErrorFunction = errorFunction;
            State = TrainerState.Ready;
        }

        public void SetTrainData(Matrix X, Vector y)
        {
            CheckTraningData(X, y);
            this.X = X;
            this.y = y;
        }

        public void Start()
        {
            if (State == TrainerState.Training)
                return;
            if (State == TrainerState.Paused)
                Continue();
            else
            {
                StepCounter = 0;
                Started?.Invoke(this, null);

                TrainFunction();
            }
        }

        public void StartTrain(Matrix X, Vector y)
        {
            if (State == TrainerState.Training)
                return;

            SetTrainData(X, y);

            StepCounter = 0;
            Started?.Invoke(this, null);

            TrainFunction();
        }

        private void TrainFunction()
        {
            State = TrainerState.Training;

            Matrix batchX;
            Vector batchY;
            var error = double.MaxValue;

            while (State == TrainerState.Training)
            {
                try
                {
                    var batchStart = StepCounter % X.Rows;
                    (batchX, batchY) = GetBatchX(X, y, batchStart, BatchSize);
                    Machine.Step(batchX, batchY);


                    StepCounter++;
                    if (StepCounter % NotifySteps == 0)
                        Notify?.Invoke(this, new NotifyEventArgs
                        {
                            Machine = Machine,
                            X = X,
                            Y = y,
                            BatchX = batchX,
                            BatchY = batchY,
                            Step = StepCounter,
                        });

                    if (StepCounter >= MaxStep)
                        State = TrainerState.MaxStepsStopped;

                    if (ErrorLimit > 0 && ErrorFunction != null)
                    {
                        var yHat = Machine.Predict(batchX);
                        error = ErrorFunction(yHat, batchY);
                        if (error <= ErrorLimit)
                            State = TrainerState.ErrorLimitStopped;
                    }
                }
                catch(Exception e)
                {
                    if (ErrorOut != null)
                    {
                        ErrorOut(this, null);
                    }
                    else
                        throw e;
                }

                Thread.Sleep(StepWait);
            }

            if (State == TrainerState.Paused)
                Paused?.Invoke(this, null);
            else
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

            Continued?.Invoke(this, null);
            TrainFunction();
        }

        public void Stop()
        {
            if (State == TrainerState.Training)
                State = TrainerState.UserStopped;
            else if (State == TrainerState.Paused)
            {
                State = TrainerState.UserStopped;
                Notify?.Invoke(this, new NotifyEventArgs
                {
                    Machine = Machine,
                    Step = StepCounter,
                    X = X,
                    Y = y,
                });
            }
            else
                return;
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

    public class NotifyEventArgs:EventArgs
    {
        public IMachine Machine { get; internal set; }
        public Matrix X { get; internal set; }
        public Matrix BatchX { get; internal set; }
        public Vector Y { get; internal set; }
        public Vector BatchY { get; internal set; }
        public int Step { get; internal set; }
    }

    public enum TrainerState
    {
        Training,
        Ready,
        Paused,
        MaxStepsStopped,
        ErrorLimitStopped,
        UserStopped,
        ErrorStopped,
    }
}
