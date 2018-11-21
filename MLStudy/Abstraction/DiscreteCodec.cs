/*
 * Description:离散编码，用于多分类问题
 * Author:YunXiao An
 * Date:2015.11.20
 */


using System.Collections.Generic;
using System.Linq;

namespace MLStudy.Abstraction
{
    public abstract class DiscreteCodec<T>
    {
        protected List<T> categories;

        public DiscreteCodec(IEnumerable<T> categories)
        {
            this.categories = categories.Distinct().ToList();
        }

        public abstract Tensor Encode(List<T> data);
        public abstract List<T> Decode(Tensor t);
    }
}
