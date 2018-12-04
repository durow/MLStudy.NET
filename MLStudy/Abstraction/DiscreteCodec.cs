/*
 * Description:离散编码，用于多分类问题
 * Author:YunXiao An
 * Date:2015.11.20
 */


using System.Collections.Generic;
using System.Linq;

namespace MLStudy.Abstraction
{
    public abstract class DiscreteCodec
    {
        public List<string> Categories { get; private set; }

        public DiscreteCodec(IEnumerable<string> categories)
        {
            Categories = categories.Distinct().ToList();
        }

        public DiscreteCodec(IEnumerable<double> categories)
            : this(categories.Distinct().Select(a => a.ToString()))
        {
        }

        public TensorOld Encode(IEnumerable<double> data)
        {
            return Encode(data.Select(a => a.ToString()));
        }

        public abstract TensorOld Encode(IEnumerable<string> data);
        public abstract List<string> Decode(TensorOld t);
    }
}
