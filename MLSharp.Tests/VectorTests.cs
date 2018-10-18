using System;
using System.Collections.Generic;
using System.Text;
using Xunit;
using MLSharp;

namespace MLSharp.Tests
{
    public class VectorTests
    {
        [Fact]
        public void ToStringTest()
        {
            var a = new Vector(1, 2, 3, 4, 5, 6);
            var expected = "[1,2,3,4,5,6]";
            var actual = a.ToString();
            Assert.Equal(expected, actual);
        }

        [Fact]
        public void ModifyTest()
        {
            var a = new Vector(1, 2, 3, 4, 5, 6);
            a[2] = 100;
            Assert.Equal(100, a[2]);
        }

        [Fact]
        public void Add()
        {
            var a = new Vector(1, 2, 3, 4, 5, 6);
            var b = 10d;
            var c = new Vector(4, 5, 6, 7, 8, 9);

            var expected12 = new Vector(11, 12, 13, 14, 15, 16);
            var expected34 = new Vector(5, 7, 9, 11, 13, 15);

            var actual1 = a + b;
            var actual2 = b + a;
            var actual3 = a + c;
            var actual4 = c + a;

            Assert.Equal(expected12, actual1);
            Assert.Equal(expected12, actual2);
            Assert.Equal(expected34, actual3);
            Assert.Equal(expected34, actual4);
        }

        [Fact]
        public void Multiple()
        {
            var a = new Vector(1, 2, 3, 4, 5, 6);
            var b = 10d;
            var c = new Vector(4, 5, 6, 7, 8, 9);

            var expected12 = new Vector(10, 20, 30, 40, 50, 60);
            var expected34 = new Vector(4, 10, 18, 28, 40, 54);

            var actual1 = a * b;
            var actual2 = b * a;
            var actual3 = a * c;
            var actual4 = c * a;

            Assert.Equal(expected12, actual1);
            Assert.Equal(expected12, actual2);
            Assert.Equal(expected34, actual3);
            Assert.Equal(expected34, actual4);
        }
    }
}
