using System.Threading.Tasks;

namespace MLStudy
{
    public partial class Tensor
    {
        /// <summary>
        /// 当前Tensor乘上d，结果保存在当前Tensor
        /// </summary>
        /// <param name="d">要乘上的值</param>
        /// <returns>当前Tensor</returns>
        public Tensor Multiple(double d)
        {
            Apply(a => a * d);
            return this;
        }

        /// <summary>
        /// 当前Tensor和t的点积，结果保存在当前Tensor，要求两个Tensor结构一致
        /// </summary>
        /// <param name="t">乘上的Tensor</param>
        /// <returns>当前Tensor</returns>
        public Tensor MultipleElementWise(Tensor t)
        {
            if (t.ElementCount == 1)
                return Multiple(t.GetValue());

            CheckShape(shape, t.shape);

            MultipleElementWise(this, t, this);
            return this;
        }

        /// <summary>
        /// Tensor每个元素乘上数字d，结果返回为新的Tensor
        /// </summary>
        /// <param name="t">Tensor</param>
        /// <param name="d">数字d</param>
        /// <returns>包含结果的新的Tensor</returns>
        public static Tensor Multiple(Tensor t, double d)
        {
            var result = t.GetSameShape();
            Multiple(t, d, result);
            return result;
        }

        /// <summary>
        /// 两个Tensor相乘，结果返回为新的Tensor。
        /// 目前仅支持scalar、vector、matrix的乘法
        /// </summary>
        /// <param name="a">Tensor1</param>
        /// <param name="b">Tensor2</param>
        /// <returns>包含结果的新的Tensor</returns>
        public static Tensor Multiple(Tensor a, Tensor b)
        {
            if (a.ElementCount == 1)
                return Multiple(b, a.GetValue());
            if (b.ElementCount == 1)
                return Multiple(a, b.GetValue());

            if (a.Rank == 1)
                a = a.Reshape(1, a.ElementCount);
            if (b.Rank == 1)
                b = b.Reshape(b.ElementCount, 1);

            CheckMultipleShape(a, b);
            var result = new Tensor(a.shape[0], b.shape[1]);
            Multiple(a, b, result);

            return result;
        }

        /// <summary>
        /// 两个Tensor的点积，结果返回为新的Tensor，要求两个Tensor结构一致
        /// </summary>
        /// <param name="a">Tensor1</param>
        /// <param name="b">Tensor2</param>
        /// <returns>包含结果的新的Tensor</returns>
        public static Tensor MultipleElementWise(Tensor a, Tensor b)
        {
            if (a.ElementCount == 1)
                return Multiple(b, a.GetValue());
            if (b.ElementCount == 1)
                return Multiple(a, b.GetValue());

            var result = a.GetSameShape();
            MultipleElementWise(a, b, result);
            return result;
        }

        /// <summary>
        /// Tensor每个元素乘上d，结果写入result
        /// </summary>
        /// <param name="t">Tensor乘数</param>
        /// <param name="d">标量乘数</param>
        /// <param name="result">结果</param>
        public static void Multiple(Tensor t, double d, Tensor result)
        {
            Apply(t, result, a => a * d);
        }

        public static void Multiple(Tensor a, Tensor b, Tensor result)
        {
            var rows = a.shape[0];
            var cols = b.shape[1];
            var bStep = a.shape[1];

            Parallel.For(0, rows, i =>
            {
                var aStart = a.GetRawOffset(i, 0);
                var resultStart = result.GetRawOffset(i, 0);
                Parallel.For(0, cols, j =>
                {
                    var sum = 0d;
                    var bStart = b.GetRawOffset(0, j);
                    for (int k = 0; k < bStep; k++)
                    {
                        sum += a.values[aStart + k] * b.values[k * cols + j];
                    }
                    //result.values[i * cols + j] = sum;
                    result.SetValueFast(sum, i, j);
                });
            });
        }

        /// <summary>
        /// a和b的点积，结果写入result
        /// 必要的时候在调用这个方法前进行Tensor结构一致性检查
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="result"></param>
        public static void MultipleElementWise(Tensor a, Tensor b, Tensor result)
        {
            Apply(a, b, result, (x, y) => x * y);
        }

        public static Tensor operator *(Tensor t, double d)
        {
            return Multiple(t, d);
        }

        public static Tensor operator *(double d, Tensor t)
        {
            return Multiple(t, d);
        }

        public static Tensor operator *(Tensor a, Tensor b)
        {
            return Multiple(a, b);
        }
    }
}
