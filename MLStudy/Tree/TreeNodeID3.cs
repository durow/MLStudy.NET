using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;

namespace MLStudy.Tree
{
    public class TreeNodeID3 : TreeNode
    {
        public TreeNodeID3(List<DataRow> data, List<string> columns, string labelColumn, int depth, int maxDepth)
            : base(data, columns, labelColumn, depth, maxDepth)
        { }

        public override void Fork()
        {
            if (MaxDepth > 0 && Depth >= MaxDepth)
                return;

            if (CheckLabelPure(Data, LabelColumn))
                return;

            var forkDict = new Dictionary<string, Dictionary<object, List<DataRow>>>();
            var entropyDict = new Dictionary<string, double>();

            foreach (var col in AvailableColumns)
            {
                if (col == LabelColumn)
                    continue;

                var forkResult = ForkByColumn(Data, col);
                forkDict[col] = forkResult;
                entropyDict[col] = ComputeEntropy(forkResult, LabelColumn, Data.Count);
            }

            ForkColumn = entropyDict.OrderBy(k => k.Value).First().Key;

            foreach (var item in forkDict[ForkColumn])
            {
                var node = new TreeNodeID3(
                    item.Value,
                    GetNewAvailableColumns(AvailableColumns, ForkColumn),
                    LabelColumn,
                    Depth++,
                    MaxDepth
                    )
                {
                    ForkedColumn = ForkColumn,
                    ForkedValue = item.Key
                };

                ChildNodes.Add(node);
            }

            foreach (var child in ChildNodes)
            {
                child.Fork();
            }
        }

        public override bool IsInNode(DataRow row)
        {
            if (string.IsNullOrEmpty(ForkedColumn))
                return true;

            var val = row[ForkedColumn];
            if (ForkedValue == val)
                return true;
            else
                return false;
        }

        public override TreeNode Predict(DataRow row)
        {
            if (IsLeaf)
                return this;

            foreach (var item in ChildNodes)
            {
                if (item.IsInNode(row))
                    return item.Predict(row);
            }

            return this;// or throw an exeption
        }
    }
}
