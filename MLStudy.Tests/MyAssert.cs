using System;
using System.Collections.Generic;
using System.Text;
using Xunit;


namespace MLStudy.Tests
{
    public class MyAssert
    {
        public static void ApproximatelyEqual(double a, double b, double error=0.00001)
        {
            Assert.True(Math.Abs(a - b) < error);
        }
    }
}
