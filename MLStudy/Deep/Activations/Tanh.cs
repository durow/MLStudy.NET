

namespace MLStudy.Deep
{
    public class Tanh : Activations.Activation
    {
        public override Tensor PrepareTrain(Tensor input)
        {
            BackwardOutput = input.GetSameShape();
            ForwardOutput = input.GetSameShape();
            Derivative = input.GetSameShape();
            return ForwardOutput;
        }

        public override Tensor Forward(Tensor input)
        {
            Tensor.Apply(input, ForwardOutput, Functions.Tanh);
            return ForwardOutput;
        }

        public override Tensor Backward(Tensor error)
        {
            Tensor.Apply(ForwardOutput, Derivative, Derivatives.TanhFromOutput);
            Tensor.MultipleElementWise(Derivative, error, BackwardOutput);
            return BackwardOutput;
        }
    }
}
