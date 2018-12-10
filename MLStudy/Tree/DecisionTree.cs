using System;
using System.Collections.Generic;
using System.Data;
using System.Text;

namespace MLStudy.Tree
{
    public class DecisionTree
    {
        public int MaxDepth { get; set; }

        public TreeNode Root;

        public DecisionTree(TreeAlgorithm alg, int maxDepth = 0)
        {
            MaxDepth = maxDepth;
        }

        public void Grow(DataTable data, string labelName)
        {
            //var columns = 
        }
    }
}
