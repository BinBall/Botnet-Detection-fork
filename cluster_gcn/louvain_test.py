import time
import json

import community
import numpy as np
from community import community_louvain
from networkx.readwrite import json_graph
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx

from train import load_data
import tensorflow as tf
from tensorflow.compat.v1 import gfile


# # load the karate club graph
# G = nx.karate_club_graph()
#
# # compute the best partition
# partition = community_louvain.best_partition(G)
#
#
# # draw the graph
# pos = nx.spring_layout(G)
# # color the nodes according to their partition
# cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
# nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
#                        cmap=cmap, node_color=list(partition.values()))
# nx.draw_networkx_edges(G, pos, alpha=0.5)
# plt.show()
def load_graphsage_data(dataset_path, dataset_str):
    """
    <train_prefix>-G.json -- 一个networkx指定的描述输入图的json文件。节点有'val'和'test'属性，分别指定它们是否是验证集和测试集的一部分。
    <train_prefix>-id_map.json -- 一个json存储的字典，将图的节点id映射为连续的整数。
    <train_prefix>-class_map.json -- 一个存储在json中的字典，将图形节点的id映射到类。
    <train_prefix>-feats.npy [可选] --- 一个由numpy存储的节点特征数组；排序由id_map.json给出。可以省略，只有身份特征会被使用。
    <train_prefix>-walks.txt [可选] --- 一个文本文件，指定随机行走的共同出现（每行一对）（*仅用于graphsage的无监督版本）。
    """
    """Load GraphSAGE data."""
    start_time = time.time()
    tf.compat.v1.logging.info('Start initializing the Graph...')
    # 载入JSON格式的图数据文件，得到Graph的JSON对象
    graph_json = json.load(gfile.Open('{}/{}/{}-G.json'.format(dataset_path, dataset_str, dataset_str)))
    # 将Graph的JSON对象转为Networkx.Graph对象
    graph_nx = json_graph.node_link_graph(graph_json)
    tf.compat.v1.logging.info(f'Graph Initializing finished, {time.time() - start_time}')
    return graph_nx


def test():
    # Load data
    # (train_adj, full_adj, train_feats, test_feats, y_train, y_val, y_test,
    #  train_mask, val_mask, test_mask, _, val_data, test_data, num_data,
    #  visible_data) = load_data('./data/', 'ppi', True)
    #
    # tf.compat.v1.logging.info('Start initializing the Graph...')
    # start_time = time.time()
    # G = nx.MultiGraph(train_adj)
    # tf.compat.v1.logging.info(f'Graph Initializing finished, {time.time()-start_time}')
    G = load_graphsage_data('./data/', 'ppi')
    tf.compat.v1.logging.info("Start Louvain Partition...")
    start_time = time.time()
    partition = community_louvain.best_partition(G)
    tf.compat.v1.logging.info(f"Louvain Partition Finished, {time.time() - start_time}")

    modularity = community.modularity(partition, G)
    print(f"The modularity of the partition is {modularity}")

    print(partition)

test()