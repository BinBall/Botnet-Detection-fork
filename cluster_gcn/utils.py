# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Collections of preprocessing functions for different graph formats."""

import json
import time

import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import partition_utils
import scipy.sparse as sp
import sklearn.metrics
import sklearn.preprocessing
import tensorflow as tf
import h5py
from tensorflow.compat.v1 import gfile


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in gfile.Open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def sym_normalize_adj(adj):
    """Normalization by D^{-1/2} (A+I) D^{-1/2}."""
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1)) + 1e-20
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt, 0)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj


def normalize_adj(adj):
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = 1.0 / (np.maximum(1.0, rowsum))
    d_mat_inv = sp.diags(d_inv, 0)
    adj = d_mat_inv.dot(adj)
    return adj


def normalize_adj_diag_enhance(adj, diag_lambda):
    """Normalization by  A'=(D+I)^{-1}(A+I), A'=A'+lambda*diag(A')."""
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = 1.0 / (rowsum + 1e-20)
    d_mat_inv = sp.diags(d_inv, 0)
    adj = d_mat_inv.dot(adj)
    adj = adj + diag_lambda * sp.diags(adj.diagonal(), 0)
    return adj


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def calc_f1(y_pred, y_true, multilabel):
    if multilabel:
        y_pred[y_pred > 0] = 1
        y_pred[y_pred <= 0] = 0
    else:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    return sklearn.metrics.f1_score(
        y_true, y_pred, average='micro'), sklearn.metrics.f1_score(
        y_true, y_pred, average='macro')


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['num_features_nonzero']: features[0].shape})
    # feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def preprocess_multicluster(adj,
                            parts,
                            features,
                            y_train,
                            train_mask,
                            num_clusters,
                            block_size,
                            diag_lambda=-1):
    """Generate the batch for multiple clusters."""

    features_batches = []
    support_batches = []
    y_train_batches = []
    train_mask_batches = []
    total_nnz = 0
    np.random.shuffle(parts)
    for _, st in enumerate(range(0, num_clusters, block_size)):
        pt = parts[st]
        for pt_idx in range(st + 1, min(st + block_size, num_clusters)):
            pt = np.concatenate((pt, parts[pt_idx]), axis=0)
        features_batches.append(features[pt, :])
        y_train_batches.append(y_train[pt, :])
        support_now = adj[pt, :][:, pt]
        if diag_lambda == -1:
            support_batches.append(sparse_to_tuple(normalize_adj(support_now)))
        else:
            support_batches.append(
                sparse_to_tuple(normalize_adj_diag_enhance(support_now, diag_lambda)))
        total_nnz += support_now.count_nonzero()

        train_pt = []
        for newidx, idx in enumerate(pt):
            if train_mask[idx]:
                train_pt.append(newidx)
        train_mask_batches.append(sample_mask(train_pt, len(pt)))
    return (features_batches, support_batches, y_train_batches,
            train_mask_batches)


def preprocess(adj,
               features,
               y_train,
               train_mask,
               visible_data,
               num_clusters,
               diag_lambda=-1,
               partition_method='Metis', graph=None):
    """Do graph partitioning and preprocessing for SGD training."""

    # 图聚类
    if partition_method == "Louvain":
        part_adj, parts = partition_utils.louvain_partition_graph(graph, adj, visible_data)
    else:
        part_adj, parts = partition_utils.metis_partition_graph(adj, visible_data, num_clusters)

    parts = [np.array(pt, dtype=np.int32) for pt in parts if len(pt) > 0]
    # 邻接矩阵正则化
    if diag_lambda == -1:
        part_adj = normalize_adj(part_adj)
    else:
        part_adj = normalize_adj_diag_enhance(part_adj, diag_lambda)

    features_batches = []
    support_batches = []
    y_train_batches = []
    train_mask_batches = []
    total_nnz = 0
    for pt in parts:
        features_batches.append(features[pt, :])
        now_part = part_adj[pt, :][:, pt]
        total_nnz += now_part.count_nonzero()
        support_batches.append(sparse_to_tuple(now_part))
        y_train_batches.append(y_train[pt, :])

        train_pt = []
        for new_idx, idx in enumerate(pt):
            if train_mask[idx]:
                train_pt.append(new_idx)
        train_mask_batches.append(sample_mask(train_pt, len(pt)))
    return parts, features_batches, support_batches, y_train_batches, train_mask_batches


def load_botnet_data(partition_method='Metis'):
    import deepdish as dd
    start_time = time.time()
    # 从文件读取数据集
    print('Start load data...')
    train_data = dd.io.load('data/botnet/processed/c2_train.hdf5')['0']
    edge_idx = train_data["edge_index"]
    print(f'Data loaded({time.time() - start_time:.2f} s), start preprocessing...')
    start_time = time.time()
    # 构造边集
    if partition_method == 'Metis':
        # Metis不支持自连接
        edges = [[edge_idx[0][i], edge_idx[1][i]] for i, _ in enumerate(edge_idx[0]) if
                 not edge_idx[0][i] == edge_idx[1][i]]
    else:
        edges = [[edge_idx[0][i], edge_idx[1][i]] for i, _ in enumerate(edge_idx[0])]
    # 用边集构造图
    full_graph = nx.Graph(edges)
    # 获取图中节点数
    num_data = len(full_graph)
    # 将边集转为Numpy, 用于构造稀疏矩阵形式的邻接矩阵
    edges = np.array(edges, dtype=np.int32)
    # 构造稀疏矩阵形式的邻接矩阵
    full_adj = _construct_adj(edges, num_data)
    # 从数据集中读取特征矩阵
    feats = train_data["x"]
    # 从数据集中读取标签集
    labels = np.reshape(train_data["y"], (-1, 1))
    # 将全图节点全部归入训练集
    train_data = np.arange(0, num_data)
    train_adj = full_adj

    # 将全图归入测试集、验证集
    train_feats = test_feats = full_adj
    val_data = test_data = train_data
    return num_data, train_adj, full_adj, feats, train_feats, test_feats, \
           labels, train_data, val_data, test_data, full_graph, full_graph


# 根据边集构造稀疏矩阵形式存储的邻接矩阵
def _construct_adj(edges, num_data):
    adj = sp.csr_matrix((np.ones(
        (edges.shape[0]), dtype=np.float32), (edges[:, 0], edges[:, 1])),
        shape=(num_data, num_data))
    adj += adj.transpose()
    return adj


def load_graphsage_data(dataset_path, dataset_str, normalize=True, partition_method='Metis'):
    """
    <train_prefix>-G.json -- 一个networkx指定的描述输入图的json文件。节点有'val'和'test'属性，分别指定它们是否是验证集和测试集的一部分。
    <train_prefix>-id_map.json -- 一个json存储的字典，将图的节点id映射为连续的整数。
    <train_prefix>-class_map.json -- 一个存储在json中的字典，将图形节点的id映射到类。
    <train_prefix>-feats.npy [可选] --- 一个由numpy存储的节点特征数组；排序由id_map.json给出。可以省略，只有身份特征会被使用。
    <train_prefix>-walks.txt [可选] --- 一个文本文件，指定随机行走的共同出现（每行一对）（*仅用于graphsage的无监督版本）。
    """
    """Load GraphSAGE data."""
    start_time = time.time()
    # 载入JSON格式的图数据文件，得到Graph的JSON对象
    graph_json = json.load(gfile.Open('{}/{}/{}-G.json'.format(dataset_path, dataset_str, dataset_str)))
    # 将Graph的JSON对象转为Networkx.Graph对象
    full_graph = json_graph.node_link_graph(graph_json)
    # 加载Graph对应的字典
    id_map = json.load(gfile.Open('{}/{}/{}-id_map.json'.format(dataset_path, dataset_str, dataset_str)))
    is_digit = list(id_map.keys())[0].isdigit()
    id_map = {(int(k) if is_digit else k): int(v) for k, v in id_map.items()}
    print(f'len(full_graph)={len(full_graph)},len(id_map)={len(id_map)}')
    # print(len(id_map))
    # todo len(id_map)!=len(full_graph)
    # id_map = {(int(k) if is_digit else k): int(v) for k, v in id_map.items() if full_graph.has_node(k)}
    # print(len(id_map))
    # exit(0)

    class_map = json.load(gfile.Open('{}/{}/{}-class_map.json'.format(dataset_path, dataset_str, dataset_str)))
    is_instance = isinstance(list(class_map.values())[0], list)
    class_map = {(int(k) if is_digit else k): (v if is_instance else int(v)) for k, v in class_map.items()}
    # 消除无意义的点
    broken_count = 0
    to_remove = []
    for node in full_graph.nodes():
        if node not in id_map:
            to_remove.append(node)
            broken_count += 1
    if broken_count > 0:
        nx.Graph().remove_nodes_from(to_remove)
        tf.compat.v1.logging.info('Removed %d nodes that lacked proper annotations due to networkx versioning issues',
                                  broken_count)
    # 加载特征矩阵
    feats = np.load(
        gfile.Open('{}/{}/{}-feats.npy'.format(dataset_path, dataset_str, dataset_str), 'rb')).astype(np.float32)
    tf.compat.v1.logging.info(f'Loaded data ({time.time() - start_time:.2f} seconds).. now preprocessing..')
    start_time = time.time()
    # 根据图生成边表
    edges = []
    for edge in full_graph.edges():
        if edge[0] in id_map and edge[1] in id_map:
            edges.append((id_map[edge[0]], id_map[edge[1]]))
    edges = np.array(edges, dtype=np.int32)
    # 获取图中节点数量
    # todo len(id_map)!=len(full_graph)
    num_data = len(id_map)
    # print(f'len(id_map)={len(id_map)}, len(full_graph)={len(full_graph)}')
    # 从图中提取出验证集和测试集
    # todo len(id_map)!=len(full_graph)
    val_data = np.array([id_map[n] for n in full_graph.nodes() if full_graph.node[n]['val']], dtype=np.int32)
    test_data = np.array([id_map[n] for n in full_graph.nodes() if full_graph.node[n]['test']], dtype=np.int32)
    # val_data = np.array([id_map[n] for n in full_graph.nodes() if full_graph.node[n]['val'] and full_graph.has_node(n)], dtype=np.int32)
    # test_data = np.array([id_map[n] for n in full_graph.nodes() if full_graph.node[n]['test'] and full_graph.has_node(n)], dtype=np.int32)
    # 从图中提取出训练集
    is_train = np.ones(num_data, dtype=np.bool)
    is_train[val_data] = False
    is_train[test_data] = False
    # train_data记录了属于训练集的节点索引
    train_data = np.array([n for n in range(num_data) if is_train[n]], dtype=np.int32)
    train_edges = np.array([(e[0], e[1]) for e in edges if is_train[e[0]] and is_train[e[1]]], dtype=np.int32)

    # todo 根据任务为多标签和分类，处理标签
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
        labels = np.zeros((num_data, num_classes), dtype=np.float32)
        for k in class_map.keys():
            labels[id_map[k], :] = np.array(class_map[k])
    else:
        num_classes = len(set(class_map.values()))
        labels = np.zeros((num_data, num_classes), dtype=np.float32)
        for k in class_map.keys():
            labels[id_map[k], class_map[k]] = 1
    # todo 正则化
    if normalize:
        train_ids = np.array([
            id_map[n]
            for n in full_graph.nodes()
            if not full_graph.node[n]['val'] and not full_graph.node[n]['test']
        ])
        train_feats = feats[train_ids]
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)

    train_adj = _construct_adj(train_edges, num_data)
    full_adj = _construct_adj(edges, num_data)

    train_feats = feats[train_data]
    test_feats = feats

    tf.compat.v1.logging.info(f'Data loaded, {time.time() - start_time:.2f} seconds.')
    if partition_method == 'Louvain':
        tf.compat.v1.logging.info(f'num_data={num_data}, graph contains {len(full_graph)} nodes')
        # train_graph = full_graph.copy()
        # start_time = time.time()
        # train_graph = nx.from_scipy_sparse_matrix(train_adj)
        # train_graph.name = 'train_graph'
        # full_graph.name = 'full_graph'
        # print(f'Constructing full_graph uses {time.time()-start_time:.2f} seconds')
        # print(f'{train_graph.name} contains {len(train_graph.nodes())} nodes, {len(train_graph.edges())} edges')
        # print(f'{full_graph.name} contains {len(full_graph.nodes())} nodes, {len(full_graph.edges())} edges')
    else:
        train_graph = None
        full_graph = None
    print(f'shape(feats)={np.shape(feats)},shape(labels)={np.shape(labels)},shape(train_data)={np.shape(train_data)},')
    print(f'type(feats)={type(feats[0])},type(labels)={type(labels[0])},type(train_data)={type(train_data)}')
    print(f'type(feats)={type(feats)},type(labels)={type(labels)},type(train_data)={type(train_data)}')

    exit(0)
    return num_data, train_adj, full_adj, feats, train_feats, test_feats, \
           labels, train_data, val_data, test_data, full_graph, full_graph
