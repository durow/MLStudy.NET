using System;
using System.Collections.Generic;
using System.Text;
using Xunit;


namespace MLStudy.Tests
{
    public class MyAssert
    {
        public static void ApproximatelyEqual(double a, double b, double error=0.000001)
        {
            Assert.True(Math.Abs(a - b) < error);
        }

        public static void ApproximatelyEqual(Tensor a, Tensor b, double allowError = 0.000001)
        {
            Assert.True(Tensor.CheckShapeBool(a, b));

            var dataA = a.GetRawValues();
            var dataB = b.GetRawValues();
            for (int i = 0; i < a.ElementCount; i++)
            {
                Assert.True(Math.Abs(dataA[i] - dataB[i]) < allowError);
            }
        }
    }
}
