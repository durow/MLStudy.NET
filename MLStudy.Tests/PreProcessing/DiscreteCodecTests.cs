using MLStudy.PreProcessing;
using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace MLStudy.Tests.PreProcessing
{
    public class DiscreteCodecTests
    {
        DiscreteCodec<string> oh = new DiscreteCodec<string>(new List<string>
            {
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "f",
                "c",
                "a"
            });

        [Fact]
        public void OneHotTest()
        {
            Assert.Equal(6, oh.OneHotLength);

            var test = new List<string> { "a", "c", "d", "a" };
            var expected = new Tensor(new double[,]
            { { 1,0,0,0,0,0},
            { 0,0,1,0,0,0},
            { 0,0,0,1,0,0},
            { 1,0,0,0,0,0}, });
            var encode = oh.OneHotEncode(test);
            Assert.Equal(expected, encode);

            var decode = oh.OneHotDecode(encode);
            Assert.Equal(decode, test);
        }

        [Fact]
        public void DummyTest()
        {
            Assert.Equal(5, oh.DummyLength);

            var test = new List<string> { "a", "c", "d", "f" };
            var expected = new Tensor(new double[,]
            { { 1,0,0,0,0},
            { 0,0,1,0,0},
            { 0,0,0,1,0},
            { 0,0,0,0,0}, });
            var encode = oh.DummyEncode(test);
            Assert.Equal(expected, encode);

            var decode = oh.DummyDecode(encode);
            Assert.Equal(decode, test);
        }

        [Fact]
        public void MapTest()
        {
            oh.MapStart = 10;
            oh.MapStep = 5;
            var test = new List<string> { "a", "c", "d", "f" };
            var expected = new Tensor(new double[] { 10, 20, 25, 35 }, 4, 1);
            var encode = oh.MapEncode(test);
            Assert.Equal(expected, encode);

            var decode = oh.MapDecode(encode);
            Assert.Equal(decode, test);
        }
    }
}
