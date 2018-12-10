using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;

namespace MLStudy.Tree
{
    public abstract class TreeNode
    {
        public bool OutputDetail { get; set; }
        public List<TreeNode> ChildNodes { get; protected set; } = new List<TreeNode>();
        public List<DataRow> Data { get; protected set; }
        public List<string> AvailableColumns { get; private set; }
        public string LabelColumn { get; private set; }
        public string ForkColumn { get; protected set; }
        public string ForkedColumn { get; protected set; }
        public object ForkedValue { get; protected set; }
        public int Depth { get; protected set; }
        public int MaxDepth { get; protected set; }
        public bool IsLeaf { get { return ChildNodes.Count > 0 ? true : false; } }

        public TreeNode(List<DataRow> data, List<string> columns, string labelColumn, int depth, int maxDepth)
        {
            Data = data;
            AvailableColumns = columns;
            LabelColumn = labelColumn;
            Depth = depth;
            MaxDepth = maxDepth;
        }

        //check if the row is in this node
        public abstract bool IsInNode(DataRow row);

        //fork child nodes
        public abstract void Fork();

        //predict witch Node is this row belongs
        public abstract TreeNode Predict(DataRow row);

        //compute entropy if fork by this column
        public static double ComputeEntropy(Dictionary<object, List<DataRow>> forkResult, string labelColumn, int totalCount)
        {
            var result = 0d;

            foreach (var item in forkResult)
            {
                var per = item.Value.Count / totalCount;
                var entropy = ComputeEntropy(item.Value, labelColumn);
                result += per * entropy;
            }

            return result;
        }

        //compute entropy of forked child
        public static double ComputeEntropy(List<DataRow> childData, string labelColumn)
        {
            var counter = new Dictionary<object, double>();

            foreach (var row in childData)
            {
                var val = row[labelColumn];
                if (!counter.ContainsKey(val))
                    counter[val] = 1;
                else
                    counter[val]++;
            }
            var dist = counter.Select(i => i.Value / childData.Count);

            return Functions.Entropy(dist);
        }

        //fork by this column
        public static Dictionary<object, List<DataRow>> ForkByColumn(List<DataRow> data, string column)
        {
            var result = new Dictionary<object, List<DataRow>>();

            foreach (var row in data)
            {
                var val = row[column];
                if (!result.ContainsKey(val))
                    result[val] = new List<DataRow>();
                result[val].Add(row);
            }

            return result;
        }

        //return this columns except the forked column
        public static List<string> GetNewAvailableColumns(List<string> columns, string forkColumn)
        {
            var result = new List<string>();

            foreach (var item in columns)
            {
                if (item != forkColumn)
                    result.Add(item);
            }

            return result;
        }

        //check if the labels are pure
        public static bool CheckLabelPure(List<DataRow> data, string labelName)
        {
            var value = data[0][labelName];

            for (int i = 1; i < data.Count; i++)
            {
                if (value != data[i][labelName])
                    return false;
            }

            return true;
        }

        public static TreeNode GetRootTreeNode(TreeAlgorithm alg, List<DataRow> data, List<string> columns, string labelColumn,int maxDepth)
        {
            if (alg == TreeAlgorithm.ID3)
                return new TreeNodeID3(data, columns, labelColumn, 0, maxDepth);

            throw new NotImplementedException($"{alg} is not implemented yet!");
        }
    }
}
