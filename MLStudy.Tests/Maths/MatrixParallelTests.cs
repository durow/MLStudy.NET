using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace MLStudy.Tests.Maths
{
    public class MatrixParallelTests
    {
        MatrixParallel op = new MatrixParallel();
        int arow =20, acolumn = 300, bcolumn = 20;
        
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
            var expected1 = new Matrix(new double[,]{
                { 58, 64 },
                { 139, 154 }});
            var expected2 = new Matrix(new double[,]{
                { 39, 54, 69 },
                { 49, 68, 87 },
                { 59, 82, 105 }});

            var actual1 = op.Multiple(a,b);
            var actual2 = op.Multiple(b,a);

            Assert.Equal(expected1, actual1);
            Assert.Equal(expected2, actual2);
        }

        [Fact]
        public void MultipleMatrixMatrixTurbo()
        {
            var a = new Matrix(new double[,]{
                { 1, 2, 3 },
                { 4, 5, 6 }});
            var b = new Matrix(new double[,]{
                { 7, 8 },
                { 9, 10 },
                { 11, 12 } });
            var expected1 = new Matrix(new double[,]{
                { 58, 64 },
                { 139, 154 }});
            var expected2 = new Matrix(new double[,]{
                { 39, 54, 69 },
                { 49, 68, 87 },
                { 59, 82, 105 }});

            var actual1 = op.MultipleTurbo(a, b);
            var actual2 = op.MultipleTurbo(b, a);

            Assert.Equal(expected1, actual1);
            Assert.Equal(expected2, actual2);
        }

        [Fact]
        public void PerformanceSequence()
        {
            var (a, b) = GetTestMatrix();
            var r = Tensor.Multiple(a, b);
        }

        [Fact]
        public void PerformanceParallel()
        {
            var (a, b) = GetTestMatrix();
            var r = op.Multiple(a, b);
        }

        [Fact]
        public void PerformanceParallelTurbo()
        {
            var (a, b) = GetTestMatrix();
            var r = op.MultipleTurbo(a, b);
        }

        [Fact]
        public void PerformanceParallelFinal()
        {
            var (a, b) = GetTestMatrix();
            var r = op.MultipleThreadLimit(a, b, 4);
        }

        [Fact]
        public void PerformanceParallelForEach()
        {
            var (a, b) = GetTestMatrix();
            var r = op.MultipleForeach(a, b);
        }

        private (Matrix a, Matrix b) GetTestMatrix()
        {
            var a = DataEmulator.Instance.RandomMatrix(arow, acolumn);
            var b = DataEmulator.Instance.RandomMatrix(acolumn, bcolumn);
            return (a, b);
        }
    }
}
