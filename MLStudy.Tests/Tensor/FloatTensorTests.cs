using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace MLStudy.Tests
{
    public class FloatTensorTests
    {
        [Fact]
        public void CreateTest()
        {
            var a = Tensor.Empty(2, 3);
            Assert.Equal(2, a.Shape[0]);
            Assert.Equal(3, a.Shape[1]);
            Assert.Equal(2, a.Rank);
            Assert.Equal(6, a.Count);
            Assert.Equal(0, a.Values[0,2]);

            var a1 = a[1];
            Assert.Equal(3, a1.Shape[0]);
            Assert.Equal(1, a1.Rank);
            Assert.Equal(3, a1.Count);
            Assert.Equal(0, a1.Values[0]);

            a1.Values[0] = 12.3f;
            Assert.Equal(12.3f, a.Values[1, 0]);
        }
    }
}
