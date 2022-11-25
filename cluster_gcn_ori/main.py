import time

from community import community_louvain
import torch
from parsor import parameter_parser
from clustering import ClusteringMachine
from clustergcn import ClusterGCNTrainer
from utils import tab_printer, graph_reader, feature_reader, target_reader
import deepdish as dd
import networkx as nx
import numpy as np


def main():
    """
    Parsing command line parameters, reading data, graph decomposition, fitting a ClusterGCN and scoring the model.
    """
    # 载入参数
    args = parameter_parser()
    # 为PyTorch生成随机数设定种子
    torch.manual_seed(args.seed)
    # 将参数以列表格式输出
    tab_printer(args)
    # # 读入图G
    # graph = graph_reader(args.edge_path)
    # # 读入特征矩阵X
    # features = feature_reader(args.features_path)
    # # 读入目标矩阵Y
    # target = target_reader(args.target_path)
    start_time = time.time()
    print('Start load data...')
    train_data = dd.io.load('data/botnet/processed/c2_train.hdf5')['0']
    edge_idx = train_data["edge_index"]
    print(f'Data loaded({time.time()-start_time:.2f} s), start preprocessing...')
    start_time = time.time()
    edges = [[edge_idx[0][i], edge_idx[1][i]] for i, _ in enumerate(edge_idx[0])]
    # # Metis不支持自连接
    # edges = [[edge_idx[0][i], edge_idx[1][i]] for i, _ in enumerate(edge_idx[0]) if
    #          not edge_idx[0][i] == edge_idx[1][i]]
    graph = nx.Graph(edges)
    #print(f'Graph build finished({time.time()-start_time:.2f} s), start partition...')
    #start_time = time.time()
    #groups = community_louvain.best_partition(graph)
    #print(f'Partition finished({time.time() - start_time:.2f} s), start calculating modularity...')
    #start_time = time.time()
    #print(f'num_clusters={len(set(groups.values()))}, modularity={community_louvain.modularity(groups, graph)}, {time.time()-start_time:.2f} s')

    adj_view = graph.adj
    full_adj = [np.array([i[0] for i in adj_view[idx].items()], dtype=int) for idx in range(len(adj_view))]
    print(full_adj[0])
    graph = nx.MultiGraph(edges)

    exit(0)
    num_edges = 0
    for i in range(len(full_adj)):
        num_edges += len(full_adj[i])
    print(graph)
    print(f'num_edges={num_edges}, len(full_adj)={len(full_adj)}, len(full_adj[0])={len(full_adj[0])}')
    features = train_data["x"]
    target = np.reshape(train_data["y"], (-1, 1))
    # 图分解，划分特征和目标
    clustering_machine = ClusteringMachine(args, graph, features, target)
    clustering_machine.decompose()
    # 训练模型
    gcn_trainer = ClusterGCNTrainer(args, clustering_machine)

    gcn_trainer.train()
    # 预测
    gcn_trainer.test()


if __name__ == "__main__":
    main()
