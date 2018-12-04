using MLStudy.PreProcessing;
using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace MLStudy.Tests.PreProcessing
{
    public class DiscreteCodecTests
    {
        List<string> data = new List<string>
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
        };

        [Fact]
        public void OneHotTest()
        {
            var oh = new OneHotCodec(data);
            Assert.Equal(6, oh.Length);

            var test = new List<string> { "a", "c", "d", "a" };
            var expected = new TensorOld(new double[,]
            { { 1,0,0,0,0,0},
            { 0,0,1,0,0,0},
            { 0,0,0,1,0,0},
            { 1,0,0,0,0,0}, });
            var encode = oh.Encode(test);
            Assert.Equal(expected, encode);

            var decode = oh.Decode(encode);
            Assert.Equal(decode, test);
        }

        [Fact]
        public void DummyTest()
        {
            var dm = new DummyCodec(data);
            Assert.Equal(5, dm.Length);

            var test = new List<string> { "a", "c", "d", "f" };
            var expected = new TensorOld(new double[,]
            { { 1,0,0,0,0},
            { 0,0,1,0,0},
            { 0,0,0,1,0},
            { 0,0,0,0,0}, });
            var encode = dm.Encode(test);
            Assert.Equal(expected, encode);

            var decode = dm.Decode(encode);
            Assert.Equal(decode, test);
        }

        [Fact]
        public void MapTest()
        {
            var oh = new MapCodec(data);
            oh.MapStart = 10;
            oh.MapStep = 5;
            var test = new List<string> { "a", "c", "d", "f" };
            var expected = new TensorOld(new double[] { 10, 20, 25, 35 }, 4, 1);
            var encode = oh.Encode(test);
            Assert.Equal(expected, encode);

            var decode = oh.Decode(encode);
            Assert.Equal(decode, test);
        }
    }
}
