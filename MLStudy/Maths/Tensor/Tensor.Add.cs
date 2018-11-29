using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public partial class Tensor
    {
        /// <summary>
        /// 把d加到当前Tensor的每个元素上
        /// </summary>
        /// <param name="d">要加的值</param>
        /// <returns>当前Tensor</returns>
        public Tensor Add(double d)
        {
            Apply(a => a + d);
            return this;
        }

        /// <summary>
        /// 把t加到当前Tensor，t和当前Tensor必须要有相同的结构
        /// </summary>
        /// <param name="t">要加的Tensor</param>
        /// <returns>当前Tensor</returns>
        public Tensor Add(Tensor t)
        {
            if (t.ElementCount == 1)
                return Add(t.GetValue());

            CheckShape(shape, t.shape);

            Add(this, t, this);
            return this;
        }

        /// <summary>
        /// 给t的每个元素加上d，结果返回为新的Tensor
        /// </summary>
        /// <param name="t">Tensor</param>
        /// <param name="d">要加上的值</param>
        /// <returns></returns>
        public static Tensor Add(Tensor t, double d)
        {
            var result = t.GetSameShape();
            Add(t, d, result);
            return result;
        }

        public static void Add(Tensor t, double d, Tensor result)
        {
            Apply(t, result, a => a + d);
        }

        /// <summary>
        /// Tensor和Tensor对应元素相加，结果返回为新的Tensor，要求两个Tensor结构相同
        /// </summary>
        /// <param name="a">Tensor</param>
        /// <param name="b">Tensor</param>
        /// <returns>相加后的结果</returns>
        public static Tensor Add(Tensor a, Tensor b)
        {
            if (a.ElementCount == 1)
                return Add(b, a.GetValue());
            if (b.ElementCount == 1)
                return Add(a, b.GetValue());

            CheckShape(a, b);
            var result = a.GetSameShape();
            Add(a, b, result);
            return result;
        }

        /// <summary>
        /// a和b相加结果写入result参数
        /// 必要的时候在调用这个方法前进行Tensor结构一致性检查
        /// </summary>
        /// <param name="a">加数1</param>
        /// <param name="b">加数2</param>
        /// <param name="result">结果</param>
        public static void Add(Tensor a, Tensor b, Tensor result)
        {
            //放弃Tensor结构的检查

            Apply(a, b, result, (x, y) => x + y);
        }

        public static Tensor operator +(Tensor t, double d)
        {
            return Add(t, d);
        }

        public static Tensor operator +(double d, Tensor t)
        {
            return Add(t, d);
        }

        public static Tensor operator +(Tensor a, Tensor b)
        {
            return Add(a, b);
        }
    }
}
