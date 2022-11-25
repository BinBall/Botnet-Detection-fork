import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm import trange

from layers import StackedGCN


class ClusterGCNTrainer(object):
    """
    Training a ClusterGCN.
    """

    def __init__(self, args, clustering_machine):
        """
        :param ags: Arguments object.
        :param clustering_machine:
        """
        self.args = args
        self.clustering_machine = clustering_machine
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 在选定设备上创建Cluster-GCN模型
        self.create_model()

    def create_model(self):
        """
        Creating a StackedGCN and transferring to CPU/GPU.
        """
        self.model = StackedGCN(self.args, self.clustering_machine.feature_count, self.clustering_machine.class_count)
        self.model = self.model.to(self.device)

    def do_forward_pass(self, cluster):
        """
        在给定簇上进行前向传播
        输入：
            cluster: 簇索引
        输出：
            average_loss: 簇上平均损失
            node_count: 结点数
        Making a forward pass with data from a given partition.
        param cluster: Cluster index.
        return average_loss: Average loss on the cluster.
        return node_count: Number of nodes.
        """
        edges = self.clustering_machine.sg_edges[cluster].to(self.device)
        # todo 宏节点
        macro_nodes = self.clustering_machine.sg_nodes[cluster].to(self.device)
        train_nodes = self.clustering_machine.sg_train_nodes[cluster].to(self.device)
        features = self.clustering_machine.sg_features[cluster].to(self.device)
        # todo squeeze()是什么
        target = self.clustering_machine.sg_targets[cluster].to(self.device).squeeze()
        predictions = self.model(edges, features)
        average_loss = F.nll_loss(predictions[train_nodes], target[train_nodes])
        node_count = train_nodes.shape[0]
        return average_loss, node_count

    # TODO 小批量随机梯度下降
    def update_average_loss(self, batch_average_loss, node_count):
        """
        Updating the average loss in the epoch.
        param batch_average_loss: Loss of the cluster.
        param node_count: Number of nodes in currently processed cluster.
        return average_loss: Average loss in the epoch.
        """
        self.accumulated_training_loss = self.accumulated_training_loss + batch_average_loss.item() * node_count
        self.node_count_seen = self.node_count_seen + node_count
        average_loss = self.accumulated_training_loss / self.node_count_seen
        return average_loss

    def do_prediction(self, cluster):
        """
        Scoring a cluster.
        :param cluster: Cluster index.
        :return prediction: Prediction matrix with probabilities.
        :return target: Target vector.
        """
        edges = self.clustering_machine.sg_edges[cluster].to(self.device)
        macro_nodes = self.clustering_machine.sg_nodes[cluster].to(self.device)
        test_nodes = self.clustering_machine.sg_test_nodes[cluster].to(self.device)
        features = self.clustering_machine.sg_features[cluster].to(self.device)
        target = self.clustering_machine.sg_targets[cluster].to(self.device).squeeze()
        target = target[test_nodes]
        prediction = self.model(edges, features)
        prediction = prediction[test_nodes, :]
        return prediction, target

    def train(self):
        """
        Training a model.
        """
        print("Training started.\n")
        # 获取epoch数
        epochs = trange(self.args.epochs, desc="Train Loss")
        # 使用Adam
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.model.train()
        for _ in epochs:
            # 随机打乱簇
            random.shuffle(self.clustering_machine.clusters)
            self.node_count_seen = 0
            self.accumulated_training_loss = 0
            is_another_iter = False
            for cluster in self.clustering_machine.clusters:
                if is_another_iter:
                    # 清除上次积累的梯度
                    self.optimizer.zero_grad()
                    is_another_iter = False
                else:
                    is_another_iter = True
                # 前向传播
                batch_average_loss, node_count = self.do_forward_pass(cluster)
                # 计算损失
                average_loss = self.update_average_loss(batch_average_loss, node_count)
                # 反向传播求梯度
                batch_average_loss.backward()
                # 更新权重
                self.optimizer.step()
            epochs.set_description("Train Loss: %g" % round(average_loss, 4))

    def test(self):
        """
        Scoring the test and printing the F-1 score.
        """
        self.model.eval()
        self.predictions = []
        self.targets = []
        for cluster in self.clustering_machine.clusters:
            prediction, target = self.do_prediction(cluster)
            self.predictions.append(prediction.cpu().detach().numpy())
            self.targets.append(target.cpu().detach().numpy())
        self.targets = np.concatenate(self.targets)
        self.predictions = np.concatenate(self.predictions).argmax(1)
        score = f1_score(self.targets, self.predictions, average="micro")
        print("\nF-1 score: {:.4f}".format(score))
