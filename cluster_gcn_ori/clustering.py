import random

import metis
import numpy as np
import torch
from sklearn.model_selection import train_test_split


class ClusteringMachine(object):
    """
    Clustering the graph, feature set and target.
    """

    def __init__(self, args, graph, features, target):
        """
        :param args: Arguments object with parameters.
        :param graph: Networkx Graph.
        :param features: Feature matrix (ndarray).
        :param target: Target vector (ndarray).
        """
        self.args = args
        self.graph = graph
        self.features = features
        self.target = target
        self._set_sizes()
        if self.args.debug:
            print(f'目标矩阵Y：{self.target.shape}')
            print(f'特征矩阵X：{self.features.shape}')

    def _set_sizes(self):
        """
        Setting the feature and class count.
        """
        # feature_count为特征维数
        self.feature_count = self.features.shape[1]
        # class_count为目标值数
        self.class_count = np.max(self.target) + 1

    # 图分解
    def decompose(self):
        """
        Decomposing the graph, partitioning the features and target, creating Torch arrays.
        """
        # 默认使用Metis随机聚类，否则使用随机划分
        if self.args.clustering_method == "metis":
            print("\nMetis graph clustering started.\n")
            self.metis_clustering()
        else:
            print("\nRandom graph clustering started.\n")
            self.random_clustering()
        self.general_data_partitioning()
        self.transfer_edges_and_nodes()

    def random_clustering(self):
        """
        Random clustering the nodes.
        """
        self.clusters = [cluster for cluster in range(self.args.cluster_number)]
        self.cluster_membership = {node: random.choice(self.clusters) for node in self.graph.nodes()}

    def metis_clustering(self):
        """
        Clustering the graph with Metis. For details see:
        """
        # todo 输入图和期望得到的簇数，输出目标函数值(?)和分区索引列表
        _, parts = metis.part_graph(self.graph, self.args.cluster_number)
        # parts是从0到cluster_number-1的list，记录每个节点对应簇
        # 但我们只需要所有簇的对应索引序号，因此先转为不重复的set再转回list
        self.clusters = list(set(parts))
        # 字典存放节点与簇的从属关系：节点-簇
        self.cluster_membership = {node: membership for node, membership in enumerate(parts)}

    def general_data_partitioning(self):
        """
        Creating data partitions and train-test splits.
        """
        self.sg_nodes = {}
        self.sg_edges = {}
        self.sg_train_nodes = {}
        self.sg_test_nodes = {}
        self.sg_features = {}
        self.sg_targets = {}
        # todo 为什么要排序
        for cluster in self.clusters:
            # 用字典将簇中节点聚集为一个子图
            subgraph = self.graph.subgraph(
                [node for node in sorted(self.graph.nodes()) if self.cluster_membership[node] == cluster])
            # 将簇中节点按簇聚集到list中
            self.sg_nodes[cluster] = [node for node in sorted(subgraph.nodes())]
            # 建立映射关系：节点-节点索引
            mapper = {node: i for i, node in enumerate(sorted(self.sg_nodes[cluster]))}
            # 建立无向图list：[节点索引0，节点索引1]
            self.sg_edges[cluster] = [[mapper[edge[0]], mapper[edge[1]]] for edge in subgraph.edges()] + [
                [mapper[edge[1]], mapper[edge[0]]] for edge in subgraph.edges()]
            # 按测试率划分训练集和测试集并排序
            self.sg_train_nodes[cluster], self.sg_test_nodes[cluster] = train_test_split(list(mapper.values()),
                                                                                         test_size=self.args.test_ratio)
            self.sg_test_nodes[cluster] = sorted(self.sg_test_nodes[cluster])
            self.sg_train_nodes[cluster] = sorted(self.sg_train_nodes[cluster])

            # 按簇划分特征矩阵和目标矩阵
            # todo features和targets具体是划分成什么样了
            self.sg_features[cluster] = self.features[self.sg_nodes[cluster], :]
            self.sg_targets[cluster] = self.target[self.sg_nodes[cluster], :]
            # if self.args.debug:
            #     print(self.sg_targets[0].shape())

    def transfer_edges_and_nodes(self):
        """
        将数据转为PyTorch格式(Tensor类型)
        Transfering the data to PyTorch format.
        """
        for cluster in self.clusters:
            self.sg_nodes[cluster] = torch.LongTensor(self.sg_nodes[cluster])
            self.sg_edges[cluster] = torch.LongTensor(self.sg_edges[cluster]).t()
            self.sg_train_nodes[cluster] = torch.LongTensor(self.sg_train_nodes[cluster])
            self.sg_test_nodes[cluster] = torch.LongTensor(self.sg_test_nodes[cluster])
            self.sg_features[cluster] = torch.FloatTensor(self.sg_features[cluster])
            self.sg_targets[cluster] = torch.LongTensor(self.sg_targets[cluster])
