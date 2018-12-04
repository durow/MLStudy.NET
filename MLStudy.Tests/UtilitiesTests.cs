using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using Xunit;

namespace MLStudy.Tests
{
    public class UtilitiesTests
    {
        [Fact]
        public void ProbToCodeTest()
        {
            var tensor = new TensorOld(new double[] { 0.001, 0.02, 0.98 });
            var expected = new TensorOld(new double[] { 0, 0, 1 });
            var actual = Utilities.ProbabilityToCode(tensor);
            Assert.Equal(expected, actual);
        }

        [Fact]
        public void GetRandomDisctinctTest()
        {
            var test = Utilities.GetRandomDistinct(100, 200, 50);
            Assert.Equal(50, test.Length);
            Assert.Equal(50, test.ToList().Distinct().Count());
        }

        [Fact]
        public void GetRandomDisctinctTest2()
        {
            var test = Utilities.GetRandomDistinct(100);
            Assert.Equal(100, test.Length);
            Assert.Equal(100, test.ToList().Distinct().Count());
            Assert.Equal(0, test.Min());
            Assert.Equal(99, test.Max());
        }

        [Fact]
        public void GetRandomDisctinctTest3()
        {
            var test = Utilities.GetRandomDistinct(120, 230);
            Assert.Equal(110, test.Length);
            Assert.Equal(110, test.ToList().Distinct().Count());
            Assert.Equal(120, test.Min());
            Assert.Equal(229, test.Max());
        }

        [Fact]
        public void SuffleListTest()
        {
            var list = new List<int>();
            for (int i = 0; i < 100; i++)
            {
                list.Add(i);
            }
            Utilities.Shuffle(list);

            Assert.Equal(100, list.Count);
        }

        [Fact]
        public void SuffleArrayTest()
        {
            var array = new int[100];
            for (int i = 0; i < 100; i++)
            {
                array[i] = i;
            }
            Utilities.Shuffle(array);

            Assert.Equal(100, array.Length);
        }

        [Fact]
        public void SuffleDataTableTest()
        {
            var table = new DataTable();

            table.Columns.Add();
            table.Columns.Add();

            for (int i = 0; i < 100; i++)
            {
                table.Rows.Add(i, $"str{i}");
            }

            Utilities.Shuffle(table);

            Assert.Equal(100, table.Rows.Count);
        }
    }
}
