using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public partial class Tensor
    {
        /// <summary>
        /// 当前Tensor的每个元素除以d，结果保存在当前Tensor
        /// </summary>
        /// <param name="d">除数</param>
        /// <returns>当前Tensor</returns>
        public Tensor Divide(double d)
        {
            Apply(a => a / d);
            return this;
        }

        /// <summary>
        /// d除以当前Tensor的每个元素，结果保存在当前Tensor
        /// </summary>
        /// <param name="d">被除数</param>
        /// <returns>当前Tensor</returns>
        public Tensor DivideBy(double d)
        {
            Apply(a => d / a);
            return this;
        }

        /// <summary>
        /// 两个Tensor对应元素相除，结果保存在当前Tensor
        /// 还不知道有什么用 ;)
        /// </summary>
        /// <param name="t">除数</param>
        /// <returns>当前Tensor</returns>
        public Tensor DivideElementWise(Tensor t)
        {
            DivideElementWise(this, t, this);
            return this;
        }

        /// <summary>
        /// Tensor的每个元素除以d，结果返回为新的Tensor
        /// </summary>
        /// <param name="t">被除数</param>
        /// <param name="d">除数</param>
        /// <returns>包含结果的新的Tensor</returns>
        public static Tensor Divide(Tensor t, double d)
        {
            var result = t.GetSameShape();
            Apply(t, result, a => a / d);
            return result;
        }

        /// <summary>
        /// 用d去除以Tensor中的每个元素，结果返回为新的Tensor
        /// </summary>
        /// <param name="d">被除数</param>
        /// <param name="t">除数</param>
        /// <returns>包含结果的新的Tensor</returns>
        public static Tensor Divide(double d, Tensor t)
        {
            var result = t.GetSameShape();
            Apply(t, result, a => d / a);
            return result;
        }

        /// <summary>
        /// 两个Tensor对应元素相处，结果返回为新的Tensor，要求两个Tensor结构一致
        /// </summary>
        /// <param name="a">被除数</param>
        /// <param name="b">除数</param>
        /// <returns>包含结果的新的Tensor</returns>
        public static Tensor DivideElementWise(Tensor a, Tensor b)
        {
            var result = a.GetSameShape();
            DivideElementWise(a, b, result);
            return result;
        }

        /// <summary>
        /// a每个元素除以b对应元素，结果写入result
        /// 必要的时候在调用这个方法前进行Tensor结构一致性检查
        /// </summary>
        /// <param name="a">被除数</param>
        /// <param name="b">除数</param>
        /// <param name="result">结果</param>
        public static void DivideElementWise(Tensor a, Tensor b, Tensor result)
        {
            Apply(a, b, result, (x, y) => x / y);
        }

        public static Tensor operator /(Tensor t, double d)
        {
            return Divide(t, d);
        }

        public static Tensor operator /(double d, Tensor t)
        {
            return Divide(d, t);
        }
    }
}
