using System;
using System.Collections.Generic;
using System.Data;
using System.Text;

namespace MLStudy.Tree
{
    public class CARTNode : TreeNode
    {
        public CARTNode(List<DataRow> data, List<string> columns, string labelColumn, int depth, int maxDepth) 
            : base(data, columns, labelColumn, depth, maxDepth)
        { }

        public override void Fork()
        {
            throw new NotImplementedException();
        }

        public override bool IsInNode(DataRow row)
        {
            throw new NotImplementedException();
        }

        public override TreeNode Predict(DataRow row)
        {
            throw new NotImplementedException();
        }
    }
}
