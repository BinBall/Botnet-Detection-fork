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

"""Collections of partitioning functions."""

import time

import community
import metis
import numpy as np
import scipy.sparse as sp
import tensorflow.compat.v1 as tf
from community import community_louvain


def louvain_partition_graph(graph, adj, idx_nodes):
    """partition a graph by Louvain."""
    start_time = time.time()
    tf.logging.info(f'Start partition, {graph.name} contains {len(graph)} nodes')
    groups = community_louvain.best_partition(graph)
    new_groups = list(groups.values())
    num_clusters = len(set(groups.values()))
    tf.logging.info(f'Partition done, groups contains {len(new_groups)} nodes')
    parts_adj, parts = partition_graph(adj, idx_nodes,
                                       start_time,
                                       num_clusters,
                                       'Louvain', new_groups)
    # start_time = time.time()
    # tf.logging.info(f'modularity={community.modularity(groups, graph)}, {time.time()-start_time:.2f} seconds')
    return parts_adj, parts


def metis_partition_graph(adj, idx_nodes, num_clusters):
    """partition a graph by METIS."""
    return partition_graph(adj, idx_nodes,
                           time.time(),
                           num_clusters)


def partition_graph(adj, index_nodes,
                    start_time,
                    num_clusters,
                    partition_method='Metis', groups=None):

    num_nodes = len(index_nodes)
    num_all_nodes = adj.shape[0]

    tf.logging.info(f'num_nodes={num_nodes}, len(index_nodes)={len(index_nodes)},num_clusters={num_clusters}, num_all_nodes={num_all_nodes}')

    neighbor_intervals = []
    neighbors = []
    edge_cnt = 0
    neighbor_intervals.append(0)
    train_adj_lil = adj[index_nodes, :][:, index_nodes].tolil()
    train_ord_map = dict()
    train_adj_lists = [[] for _ in range(num_nodes)]
    for i in range(num_nodes):
        rows = train_adj_lil[i].rows[0]
        # self-edge needs to be removed for valid format of METIS
        if i in rows and partition_method == 'Metis':
            rows.remove(i)
        train_adj_lists[i] = rows
        neighbors += rows
        edge_cnt += len(rows)
        neighbor_intervals.append(edge_cnt)
        train_ord_map[index_nodes[i]] = i

    if num_clusters > 1:
        if partition_method == 'Metis':
            _, groups = metis.part_graph(train_adj_lists, num_clusters, seed=1)
    else:
        groups = [0] * num_nodes  # TODO,cluster based on labels

    part_row = []
    part_col = []
    part_data = []
    parts = [[] for _ in range(num_clusters)]
    for node_index in range(num_nodes):
        group_index = groups[node_index]
        node_origin_index = index_nodes[node_index]
        parts[group_index].append(node_origin_index)
        for neighbor_origin_index in adj[node_origin_index].indices:
            try:
                neighbor_index = train_ord_map[neighbor_origin_index]
                if groups[neighbor_index] == group_index:
                    part_data.append(1)
                    part_row.append(node_origin_index)
                    part_col.append(neighbor_origin_index)
            except IndexError or KeyError:
                # print(f'train_ord_map={train_ord_map},len(train_ord_map)={train_ord_map}')
                print(f'num_clusters={num_clusters}, num_nodes={num_nodes}')
                print(f'node_index={node_index}, node_origin_index={node_origin_index}')
                print(f'group_index={group_index}, neighbor_index={neighbor_index}, neighbor_origin_index={neighbor_origin_index}')
                print(f'len(train_ord_map)={len(train_ord_map)}')
                print(f'len(groups)={len(groups)}')
                exit(0)
    part_data.append(0)
    part_row.append(num_all_nodes - 1)
    part_col.append(num_all_nodes - 1)
    part_adj = sp.coo_matrix((part_data, (part_row, part_col))).tocsr()

    tf.logging.info(f'{partition_method} Partition done, {time.time() - start_time:.5f} seconds.', )
    return part_adj, parts
