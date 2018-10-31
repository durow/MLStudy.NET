using System;
using System.Collections.Generic;
using System.Text;
using Xunit;
using MLStudy;

namespace MLStudy.Tests
{
    public class MatrixTests
    {
        [Fact]
        public void ToStringTest()
        {
            var a = new Matrix(new double[,]{
                { 1, 2, 3 },
                { 4, 5, 6 }});
            var expected = "[[1,2,3],\n[4,5,6]]";
            var actual = a.ToString();
            Assert.Equal(expected, actual);
        }

        [Fact]
        public void Modify()
        {
            var a = new Matrix(new double[,]{
                { 1, 2, 3 },
                { 4, 5, 6 }});
            a[1,2] = 100;
            Assert.Equal(100, a[1,2]);
        }

        [Fact]
        public void SumMean()
        {
            var a = new Matrix(new double[,]{
                { 1, 2, 3 },
                { 4, 5, 6 }});
            var expected1 = 21d;
            var actual1 = a.Sum();
            var expected2 = expected1 / 6d;
            var actual2 = a.Mean();

            Assert.Equal(expected1, actual1);
            Assert.Equal(expected2, actual2);
        }

        [Fact]
        public void Transpose()
        {
            var a = new Matrix(new double[,]{
                { 1, 2, 3 },
                { 4, 5, 6 }});
            var expected = new Matrix(new double[,]{
                { 1, 4 },
                { 2, 5 },
                { 3, 6 } });
            var actual = a.Transpose();

            Assert.Equal(expected, actual);
        }

        [Fact]
        public void AddMatrixScalar()
        {
            var a = new Matrix(new double[,]{
                { 1, 2, 3 },
                { 4, 5, 6 }});
            var b = 10d;
            var expected = new Matrix(new double[,]{
                { 11, 12, 13 },
                { 14, 15, 16 }});

            var actual1 = a + b;
            var actual2 = b + a;

            Assert.Equal(expected, actual1);
            Assert.Equal(expected, actual2);
        }

        [Fact]
        public void AddMatrixMatrix()
        {
            var a = new Matrix(new double[,]{
                { 1, 2, 3 },
                { 4, 5, 6 }});
            var b = new Matrix(new double[,]{
                { 7, 8, 9 },
                { 10, 11, 12 }});
            var expected = new Matrix(new double[,]{
                { 8, 10, 12 },
                { 14, 16, 18 }});

            var actual1 = a + b;
            var actual2 = b + a;

            Assert.Equal(expected, actual1);
            Assert.Equal(expected, actual2);
        }

        [Fact]
        public void MultipleMatrixScalar()
        {
            var a = new Matrix(new double[,]{
                { 1, 2, 3 },
                { 4, 5, 6 }});
            var b = 10d;
            var expected = new Matrix(new double[,]{
                { 10, 20, 30 },
                { 40, 50, 60 }});

            var actual1 = a * b;
            var actual2 = b * a;

            Assert.Equal(expected, actual1);
            Assert.Equal(expected, actual2);
        }

        [Fact]
        public void MultipleMatrixMatrix()
        {
            var a = new Matrix(new double[,]{
                { 1, 2, 3 },
                { 4, 5, 6 }});
            var b = new Matrix(new double[,]{
                { 7, 8 },
                { 9, 10 },
                { 11, 12 } });
            var expected1= new Matrix(new double[,]{
                { 58, 64 },
                { 139, 154 }});
            var expected2 = new Matrix(new double[,]{
                { 39, 54, 69 },
                { 49, 68, 87 },
                { 59, 82, 105 }});
            var actual1 = a * b;
            var actual2 = b * a;

            Assert.Equal(expected1, actual1);
            Assert.Equal(expected2, actual2);
        }

        [Fact]
        public void MultipleMatrixMatrix2()
        {
            var a = new Matrix(new double[,]{
                { 1 },
                { 2 },
                { 3 } });
            var b = new Matrix(new double[,]{
                { 4, 5, 6 }});

            var expected1 = new Matrix(new double[,]{
                { 4, 5, 6 },
                { 8, 10, 12 },
                { 12, 15, 18 }});

            var actual1 = a * b;

            Assert.Equal(expected1, actual1);
        }

        [Fact]
        public void MultipleMatrixVector()
        {
            var a = new Matrix(new double[,]{
                { 1, 2, 3 },
                { 4, 5, 6 }});
            var v = new Vector(7, 8, 9);
            var expected = new Matrix(new double[,]{
                { 50 },
                { 122 }});
            var actual = a * v;

            Assert.Equal(expected, actual);
        }

        [Fact]
        public void MultipleVectorMatrix()
        {
            var a = new Matrix(new double[,]{
                { 1, 2, 3 },
                { 4, 5, 6 }});
            var v = new Vector(7, 8);
            var expected = new Matrix(new double[,]{
                { 39, 54, 69 }});
            var actual = v * a;

            Assert.Equal(expected, actual);
        }

        [Fact]
        public void ApplyFunction()
        {
            var m = new Matrix(new double[,] {
            { 1,2,3}, { 4,5,6} });
            var expected = new Matrix(new double[,] {
            { 1,4,9}, { 16,25,36} });
            var actual = m.ApplyFunction(x => x * x);

            Assert.Equal(expected, actual);
        }
    }
}
