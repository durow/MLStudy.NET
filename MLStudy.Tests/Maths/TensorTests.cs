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
            var data = new double[,] { { 1, 2, 3 }, { 4, 5, 6 } };
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
        public void ReshapeTest()
        {
            var t = Tensor.Rand(100);
            var r = t.Reshape(5, 20);

            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 20; j++)
                {
                    Assert.Equal(t[i * 20 + j], r[i, j]);
                }
            }

            //测试Reshape只是视图
            r[0, 0] = 123;
            r[0, 1] = 456;
            Assert.Equal(123, t[0]);
            Assert.Equal(456, t[1]);
        }

        [Fact]
        public void ITest()
        {
            var t = Tensor.I(10);
            var shape = t.Shape;
            Assert.Equal(2, t.Rank);
            Assert.Equal(100, t.ElementCount);
            Assert.Equal(10, shape[0]);
            Assert.Equal(10, shape[1]);

            for (int i = 0; i < shape[0]; i++)
            {
                for (int j = 0; j < shape[1]; j++)
                {
                    if (i == j)
                        Assert.Equal(1, t[i, j]);
                    else
                        Assert.Equal(0, t[i, j]);
                }
            }
        }

        [Fact]
        public void AddTest()
        {
            var a = new Tensor(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var b = new Tensor(new double[,] { { 3, 2, 1 }, { 6, 5, 4 } });

            var a10 = a + 10;
            var a10r = 10 + a;
            var a10Expected = new Tensor(new double[,] { { 11, 12, 13 }, { 14, 15, 16 } });
            Assert.Equal(a10Expected, a10);
            Assert.Equal(a10Expected, a10r);

            var ab = a + b;
            var abr = b + a;
            var abExpected = new Tensor(new double[,] { { 4, 4, 4 }, { 10, 10, 10 } });
            Assert.Equal(abExpected, ab);
            Assert.Equal(abExpected, abr);
        }

        [Fact]
        public void MinusTest()
        {
            var a = new Tensor(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var b = new Tensor(new double[,] { { 3, 2, 1 }, { 6, 5, 4 } });

            var a10 = a - 10;
            var a10Expected = new Tensor(new double[,] { { -9, -8, -7 }, { -6, -5, -4 } });
            Assert.Equal(a10Expected, a10);

            var a10r = 10 - a;
            var a10rExpected = new Tensor(new double[,] { { 9, 8, 7 }, { 6, 5, 4 } });
            Assert.Equal(a10rExpected, a10r);

            var ab = a - b;
            var abExpected = new Tensor(new double[,] { { -2, 0, 2 }, { -2, 0, 2 } });
            Assert.Equal(abExpected, ab);
        }

        [Fact]
        public void MultipleTest()
        {
            var a = new Tensor(new double[] { 1, 2, 3 });
            var b = new Tensor(new double[] { 4, 5, 6 });
            var c = new Tensor(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var d = new Tensor(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
            var e = new Tensor(new double[,] { { 3, 2, 1 }, { 6, 5, 4 } });

            var ab = a * b;
            Assert.Equal(32, ab.GetValue());

            var ad = a * d;
            var adExpected = new Tensor(new double[] { 22, 28 }, 1, 2);
            Assert.Equal(adExpected, ad);

            var ca = c * a;
            var caExpected = new Tensor(new double[] { 14, 32 }, 2, 1);
            Assert.Equal(caExpected, ca);

            var abew = Tensor.MultipleElementWise(a, b);
            var abewExpected = new Tensor(new double[] { 4, 10, 18 });
            Assert.Equal(abewExpected, abew);

            var ceew = Tensor.MultipleElementWise(c, e);
            var ceewExpected = new Tensor(new double[,] { { 3, 4, 3 }, { 24, 25, 24 } });
            Assert.Equal(ceewExpected, ceew);
        }

        [Fact]
        public void TransposeTest()
        {
            var a = new Tensor(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var actual = a.Transpose();
            var expected = new Tensor(new double[,] { { 1, 4 }, { 2, 5 }, { 3, 6 } });
            Assert.Equal(expected, actual);
        }

        //[Fact]
        //public void NegativeIndexTest()
        //{
        //    var t = new Tensor(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, 3, 3);

        //    Assert.Equal(9, t[-1, -1]);
        //    Assert.Equal(8, t[-1, -2]);

        //    var test = t.GetTensorByDim1(-2);
        //    var expected = new Tensor(new double[] { 4, 5, 6 });
        //    Assert.Equal(expected, test);
        //}
    }
}
