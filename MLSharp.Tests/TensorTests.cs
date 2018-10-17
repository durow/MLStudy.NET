using System;
using Xunit;
using MLSharp;


namespace MLSharp.Tests
{
    public class TensorTests
    {
        [Fact]
        public void GetRowTest()
        {
            double[,] test = { 
                { 1, 2, 3 }, 
                { 4, 5, 6 },
                { 7, 8, 9 },
                { 10, 11, 12 }};
            double[] expected = { 10, 11, 12 };
            var actual = Tensor.GetRow(test, 3);
            for (int i = 0; i < actual.Length; i++)
            {
                Assert.Equal(expected[i], actual[i]);
            }
        }

        [Fact]
        public void GetColumnTest()
        {
            double[,] test = {
                { 1, 2, 3 },
                { 4, 5, 6 },
                { 7, 8, 9 },
                { 10, 11, 12 }};
            double[] expected = { 3, 6, 9, 12};
            var actual = Tensor.GetColumn(test, 2);
            for (int i = 0; i < actual.Length; i++)
            {
                Assert.Equal(expected[i], actual[i]);
            }
        }

        [Fact]
        public void AddTestD1()
        {
            double[] a = { 1, 2, 3, 4, 5, 6 };
            double[] b = { 7, 8, 9, 10, 11, 12 };
            double[] expected = { 8, 10, 12, 14, 16, 18 };

            var actual = Tensor.Add(a, b);

            for (int i = 0; i < actual.Length; i++)
            {
                Assert.Equal(expected[i], actual[i]);
            }
        }

        [Fact]
        public void AddTestD2()
        {
            double[,] a = { { 1, 2, 3 }, { 4, 5, 6 } };
            double[,] b = { { 11, 12, 13 }, { 14, 15, 16 } };
            double[,] expected = { { 12, 14, 16 }, { 18, 20, 22 } };

            var actual = Tensor.Add(a, b);

            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < a.GetLength(1); j++)
                {
                    Assert.Equal(expected[i, j], actual[i, j]);
                }
            }
        }
    }
}
