using System;
using Xunit;
using MLStudy;


namespace MLStudy.Tests
{
    public class TensorTests
    {
        [Fact]
        public void AddVectorScalar()
        {
            var a = new Vector(1, 2, 3, 4, 5, 6);
            var b = 10d;
            var expected = new Vector(11, 12, 13, 14, 15, 16);

            var actual = Tensor.Add(a, b);

            Assert.Equal(expected, actual);
        }

        [Fact]
        public void AddVectorVector()
        {
            var a = new Vector(1, 2, 3, 4, 5, 6);
            var b = new Vector(7, 8, 9, 10, 11, 12);
            var expected = new Vector(8, 10, 12, 14, 16, 18);

            var actual = Tensor.Add(a, b);

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

            var actual = Tensor.Add(a, b);

            Assert.Equal(expected, actual);
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

            var actual = Tensor.Add(a, b);

            Assert.Equal(expected, actual);
        }

        [Fact]
        public void MultipleVectorScalar()
        {
            var a = new Vector(1, 2, 3, 4, 5, 6);
            var b = 10d;
            var expected = new Vector(10, 20, 30, 40, 50, 60);

            var actual = Tensor.Multiple(a, b);

            Assert.Equal(expected, actual);
        }

        [Fact]
        public void MultipleVectorVector()
        {
            var a = new Vector(1, 2, 3, 4, 5, 6);
            var b = new Vector(7, 8, 9, 10, 11, 12);
            var expected = new Vector(7, 16, 27, 40, 55, 72);

            var actual = Tensor.Multiple(a, b);

            Assert.Equal(expected, actual);
        }

        [Fact]
        public void MultipleVectorVectorAsMatrix()
        {
            var a = new Vector(1, 2, 3, 4, 5, 6);
            var b = new Vector(7, 8, 9, 10, 11, 12);
            var expected = 217;

            var actual = Tensor.MultipleAsMatrix(a, b);

            Assert.Equal(expected, actual);
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

            var actual = Tensor.Multiple(a, b);

            Assert.Equal(expected, actual);
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
            var expected = new Matrix(new double[,]{
                { 58, 64 },
                { 139, 154 }});
            var actual = Tensor.Multiple(a, b);

            Assert.Equal(expected, actual);
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
            var actual = Tensor.Multiple(a, v);

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
            var actual = Tensor.Multiple(v, a);

            Assert.Equal(expected, actual);
        }
    }
}
