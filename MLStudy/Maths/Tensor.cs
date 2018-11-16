using System;
using System.Collections.Generic;
using System.Text;

namespace MLStudy
{
    public class Tensor
    {
        private float[] values;
        private int[] shape;

        /// <summary>
        /// 张量的阶
        /// </summary>
        public int Rank { get; private set; }

        /// <summary>
        /// 张量的形状
        /// </summary>
        public int[] Shape
        {
            get
            {
                return getShape();
            }
        }

        public float this[params int[] index]
        {
            get
            {
                if(index.Length != Rank)
                    throw new 
            }
            set
            {

            }
        }

        private Tensor()
        { }

        /// <summary>
        /// 创建一个张量
        /// </summary>
        /// <param name="shape">张量的形状</param>
        public Tensor(params int[] shape)
        {
            this.shape = shape;
            Rank = shape.Length;

        }

        private int[] getShape()
        {
            var result = new int[shape.Length];
            shape.CopyTo(result, 0);
            return result;
        }

        private int getTotalLength()
        {
            var result = 1;
            for (int i = 0; i < shape.Length; i++)
            {
                result *= shape[i];
            }
            return result;
        }
    }
}
