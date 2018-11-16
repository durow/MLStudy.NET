using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace MLStudy.Tests.Maths
{
    public class TensorTests
    {
        [Fact]
        public void CreateTest()
        {
            var t = new Tensor(3, 4);

            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    Assert.Equal(0, t[i, j]);
                }
            }

            Assert.Equal(3, t.Shape[0]);
            Assert.Equal(4, t.Shape[1]);
            Assert.Equal(2, t.Shape.Length);
            Assert.Equal(2, t.Rank);
            Assert.Equal(12, t.ElementCount);
        }

        [Fact]
        public void CreateTest2()
        {
            var data = new float[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            var t = new Tensor(data);

            Assert.Equal(2, t.Shape[0]);
            Assert.Equal(3, t.Shape[1]);
            Assert.Equal(2, t.Shape.Length);
            Assert.Equal(2, t.Rank);
            Assert.Equal(6, t.ElementCount);

            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    Assert.Equal(data[i, j], t[i, j]);
                }
            }
        }

        [Fact]
        public void CreateTest3()
        {
            var data = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var t = new Tensor(data, 1, 3, 3);
            Assert.Equal(1, t.Shape[0]);
            Assert.Equal(3, t.Shape[1]);
            Assert.Equal(3, t.Shape[2]);
            Assert.Equal(3, t.Shape.Length);
            Assert.Equal(3, t.Rank);
            Assert.Equal(9, t.ElementCount);

            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    Assert.Equal(data[i * 3 + j], t[0, i, j]);
                }
            }
        }

        [Fact]
        public void CreateTest4()
        {
            var t = new Tensor();
            t.SetValue(12);
            var actual = t.GetValue();
            Assert.Equal(12, actual);
        }

        [Fact]
        public void ApplyTest()
        {
            var t = Tensor.Rand(20, 30);
            var n = Tensor.Apply(t, a => a * a);

            for (int i = 0; i < 20; i++)
            {
                for (int j = 0; j < 30; j++)
                {
                    Assert.Equal(t[i, j] * t[i, j], n[i, j]);
                }
            }
        }

        [Fact]
        public void MultipleTest()
        {
            var a = new Tensor(new double[] { 1, 2, 3 });
            var b = new Tensor(new double[] { 4, 5, 6 });
            var c = new Tensor(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var d = new Tensor(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });

            var ab = a * b;
            Assert.Equal(32, ab.GetValue());

            var ad = a * d;
            var adExpected = new Tensor(new double[] { 22, 28 }, 1, 2);
            Assert.Equal(adExpected, ad);

            var ca = c * a;
            var caExpected = new Tensor(new double[] { 14, 32 }, 2, 1);
            Assert.Equal(caExpected, ca);
        }

        [Fact]
        public void TransposeTest()
        {
            var a = new Tensor(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var actual = a.Transpose();
            var expected = new Tensor(new double[,] { { 1, 4 }, { 2, 5 }, { 3, 6 } });
            Assert.Equal(expected, actual);
        }
    }
}
