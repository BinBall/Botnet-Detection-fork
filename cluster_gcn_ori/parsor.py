import argparse


def parameter_parser():
    """
    A method to parse up command line parameters. By default, it trains on the PubMed dataset.
    The default hyperparameters give a good quality representation without grid search.
    解析命令行参数方法。默认使用PubMed数据集进行训练。
    默认的超参数在没有网格搜索的情况下给出了一个良好的质量表示。
    """
    parser = argparse.ArgumentParser(description="Run .")
    # 邻接矩阵A文件存放路径
    parser.add_argument("--edge-path",
                        nargs="?",
                        default="../input/edges.csv",
                        help="Edge list csv.")
    # 特征矩阵X文件存放路径
    parser.add_argument("--features-path",
                        nargs="?",
                        default="../input/features.csv",
                        help="Features json.")
    # 目标矩阵Y文件存放路径
    parser.add_argument("--target-path",
                        nargs="?",
                        default="../input/target.csv",
                        help="Target classes csv.")
    # 聚类方法，默认Metis
    parser.add_argument("--clustering-method",
                        nargs="?",
                        default="metis",
                        help="Clustering method for graph decomposition. Default is the metis procedure.")
    # 训练epoch数，默认200
    parser.add_argument("--epochs",
                        type=int,
                        default=200,
                        help="Number of training epochs. Default is 200.")
    # PyTorch随机数种子，默认42
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for train-test split. Default is 42.")
    # Dropout率，默认0.5
    parser.add_argument("--dropout",
                        type=float,
                        default=0.5,
                        help="Dropout parameter. Default is 0.5.")
    # 学习率，默认0.01
    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.01,
                        help="Learning rate. Default is 0.01.")
    # 测试率，默认0.9
    parser.add_argument("--test-ratio",
                        type=float,
                        default=0.9,
                        help="Test data ratio. Default is 0.9.")
    # 聚类数，默认10
    parser.add_argument("--cluster-number",
                        type=int,
                        default=10,
                        help="Number of clusters extracted. Default is 10.")

    parser.add_argument("--debug", type=bool, default=False, help="Print the debug info or not. Default is False")

    parser.set_defaults(layers=[16, 16, 16])

    return parser.parse_args()
