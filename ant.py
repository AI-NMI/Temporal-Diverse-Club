import os
import h5py
import random
import numpy as np
import networkx as nx
import pandas as pd
import community as community_louvain
from infomap import Infomap
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity
from scipy.optimize import linear_sum_assignment
import copy


def participation_coef(W, ci):
    '''
    Participation coefficient is a measure of diversity of intermodular
    connections of individual nodes.
    Parameters
    ----------
    W : NxN np.ndarray
        binary/weighted directed/undirected connection matrix
    ci : Nx1 np.ndarray
        community affiliation vect[p
    Returns
    -------
    P : Nx1 np.ndarray
        participation coefficient
    '''

    _, ci = np.unique(ci, return_inverse=True)
    ci += 1
    n = len(W)
    Ko = np.sum(W, axis=1)
    Gc = np.dot((W != 0), np.diag(ci))
    Kc2 = np.zeros((n,))

    for i in range(1, int(np.max(ci)) + 1):
        Kc2 += np.square(np.sum(W * (Gc == i), axis=1))

    P = np.ones((n,)) - Kc2 / np.square(Ko)
    # P=0 if for nodes with no (out) neighbors
    P[np.where(np.logical_not(Ko))] = 0

    return P

def cal_pc(dy_mat, dy_com):
    dynamic_P=[]
    for t in range(T):
        W=dy_mat[t]
        ci=dy_com[t]
        n = len(W)  # number of vertices
        Ko = np.sum(W, axis=1)  # (out) degree加权出度
        Gc = np.dot((W != 0), np.diag(ci))  # neighbor community affiliation相邻节点的从属社区，n*n
        Kc2 = np.zeros((n,))  # community-specific neighbors

        for i in range(1, int(np.max(ci)) + 1):  # 计算某节点到所有社区（加权）平方和
            Kc2 += np.square(np.sum(W * (Gc == i), axis=1))

        P = np.ones((n,)) - Kc2 / np.square(Ko)
        # P=0 if for nodes with no (out) neighbors
        P[np.where(np.logical_not(Ko))] = 0
        dynamic_P.append(P)
    dynamic_P = np.array(dynamic_P)
    return dynamic_P

def cal_degrees(dy_mat):
    dynamic_degrees=[]
    for t in range(T):
        W=dy_mat[t]
        degrees=np.sum(W,axis=1)
        dynamic_degrees.append(degrees)
    dynamic_degrees=np.array(dynamic_degrees)
    return dynamic_degrees



def Random_Graph(G, p):
    original_nodes = list(G.nodes())
    original_edges = list(G.edges(data=True))
    degrees = dict(G.degree(weight='weight'))

    new_graph = nx.Graph()
    new_graph.add_nodes_from(original_nodes)
    new_graph.add_edges_from(original_edges)

    degree_to_remove = int(sum(degrees.values()) * p)
    removed_degree = 0

    while removed_degree < degree_to_remove:
        edge = random.choice(list(new_graph.edges(data=True)))
        current_weight = edge[2]['weight']

        if current_weight > 1:
            # reduce weight
            new_graph[edge[0]][edge[1]]['weight'] -= 1
            removed_degree += 1
        else:
            # delete edge
            new_graph.remove_edge(edge[0], edge[1])
            removed_degree += 1

    # reconnection
    while removed_degree > 0:
        current_degrees = dict(new_graph.degree(weight='weight'))
        possible_nodes = [node for node in original_nodes if current_degrees[node] < degrees[node]]
        if len(possible_nodes) < 2:
            break

        n1, n2 = random.sample(possible_nodes, 2)
        if new_graph.has_edge(n1, n2):
            new_graph[n1][n2]['weight'] += 1
            removed_degree -= 1
        else:
            new_graph.add_edge(n1, n2, weight=1)
            removed_degree -= 1

    return new_graph

def clubness(g, random_g, index_k):
    sum_g=0
    sum_random=0
    for i in index_k:
        for j in index_k:
            sum_g += g[i][j]
            sum_random += random_g[i][j]
    if sum_random==0: return 1
    return sum_g/sum_random

def temporal_DC(dy_mat, dy_ran, pc, k, delta, T):
    index_k=np.array(np.where(pc >= k)[0])
    size_k=len(index_k)
    c=np.zeros((T - delta,))
    for t in range(T - delta):
        a=np.array(dy_mat[t])
        b=np.array(dy_ran[t])
        for d in range(delta):
            for i in index_k:
                for j in index_k:
                    if dy_mat[t+d][i][j]==0:
                        a[i][j]=0
                    if dy_ran[t+d][i][j]==0:
                        b[i][j]=0
        c[t]=clubness(a,b,index_k)
    return c

def temporal_RC(dy_mat, dy_ran, degrees, k, delta, T):
    index_k = np.array(np.where(degrees >= k)[0])
    size_k = len(index_k)
    c = np.zeros((T - delta,))
    for t in range(T - delta):
        a = np.array(dy_mat[t])
        b = np.array(dy_ran[t])
        for d in range(delta):
            for i in index_k:
                for j in index_k:
                    if dy_mat[t + d][i][j] == 0:
                        a[i][j] = 0
                    if dy_ran[t + d][i][j] == 0:
                        b[i][j] = 0
        c[t] = clubness(a, b, index_k)
    return c

def align_labels(prev_labels, curr_labels):
    prev_comms = np.unique(prev_labels)
    curr_comms = np.unique(curr_labels)
    sim_matrix = np.zeros((len(prev_comms), len(curr_comms)))

    # Construct similarity matrix: number of overlapping nodes
    for i, pc in enumerate(prev_comms):
        for j, cc in enumerate(curr_comms):
            sim_matrix[i, j] = np.sum((prev_labels == pc) & (curr_labels == cc))

    # Use Hungarian algorithm to find maximum matching and remap labels
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)
    mapping = {curr_comms[col]: prev_comms[row] for row, col in zip(row_ind, col_ind)}
    new_labels = np.array([mapping.get(lab, lab) for lab in curr_labels])

    return new_labels

def filter(dynamic_graphs):
    """
    Retain a small number of high-weight edges or a large number of low-weight edges
    at each time point, ensuring the total degree sum of the two networks remain equal.

    return: Tuple of two lists of networkx Graphs (high_weight_graphs, low_weight_graphs)
    """
    heavy_graphs = []
    thin_graphs = []

    for G in dynamic_graphs:
        heavy_graph = G.copy()
        thin_graph = G.copy()

        edges = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]
        edges.sort(key=lambda x: x[2], reverse=True)

        total_degree = sum(dict(G.degree(weight="weight")).values())
        target_degree=total_degree/4

        # Select high-weight edges until the accumulated degree reaches the target threshold
        heavy_edges = []
        current_degree = 0
        for u, v, w in edges:
            heavy_edges.append((u, v))
            current_degree += w
            if current_degree >= target_degree:
                break

        # Select low-weight edges until the accumulated degree reaches the target threshold
        thin_edges = []
        current_degree = 0
        for u, v, w in reversed(edges):
            thin_edges.append((u, v))
            current_degree += w
            if current_degree >= target_degree:
                break

        for u, v in list(heavy_graph.edges()):
            if (u, v) not in heavy_edges:
                heavy_graph.remove_edge(u, v)

        for u, v in list(thin_graph.edges()):
            if (u, v) not in thin_edges:
                thin_graph.remove_edge(u, v)

        heavy_graphs.append(heavy_graph)
        thin_graphs.append(thin_graph)
    return heavy_graphs, thin_graphs

def Survival_analysis(matrix, k=80):
    '''
    Returns
    -------
    a lifespan list that contains whether a node is in club at each time point.
    '''
    club_membership = np.zeros_like(matrix, dtype=bool)

    for t in range(T):
        pc=matrix[t]
        top_nodes = np.where(pc>=np.percentile(pc, k))[0]
        club_membership[t, top_nodes] = True

    survival_list = np.sum(club_membership, axis=0)
    return survival_list

def Club_Transition(pc, degrees, k_DC=80, k_RC=80, node_class=None):
    '''
    check if there is a club transition (e.g., from TDC to TRC)

    Returns
    -------
    delay: first tenure of the clubs
    (i.e., TDC: t_0 and TRC: t_10)
    '''
    is_diverse = np.zeros((T, N), dtype=bool)
    is_rich = np.zeros((T, N), dtype=bool)

    # diverse club
    for t in range(T):
        nonzero_mask=np.where(pc > 0)[0]
        threshold_pc = np.percentile(pc[t][nonzero_mask], k_DC)
        is_diverse[t] = pc[t] >= threshold_pc

    # rich club
    for t in range(T):
        nonzero_mask=np.where(degrees > 0)[0]
        threshold_deg = np.percentile(degrees[t][nonzero_mask], k_RC)
        is_rich[t] = degrees[t] >= threshold_deg

    def find_longest_span(times):
        if len(times) == 0:
            return (np.nan, np.nan, 0)
        longest = (times[0], times[0], 1)
        current_start = times[0]
        current_len = 1

        for prev, curr in zip(times, times[1:]):
            if curr == prev + 1:
                current_len += 1
                if current_len > longest[2]:
                    longest = (current_start, curr, current_len)
            else:
                current_start = curr
                current_len = 1
        return longest

    delay=[]
    diverse_span=[]
    rich_span=[]
    for i in node_class:
        diverse_times = np.where(is_diverse[:, i])[0]
        rich_times = np.where(is_rich[:, i])[0]
        if len(diverse_times) > 0:
            t_diverse = diverse_times[0]
        else:
            t_diverse = np.nan

        if len(rich_times) > 0:
            t_rich = rich_times[0]
        else:
            t_rich = np.nan
        delay.append((i, t_diverse, t_rich))

        start_d, end_d, len_d = find_longest_span(diverse_times)
        diverse_span.append((i, start_d, end_d, len_d))

        start_r, end_r, len_r = find_longest_span(rich_times)
        rich_span.append((i, start_r, end_r, len_r))

    return delay,diverse_span,rich_span

# ant colony is a weighted dataset!!!
ant_df = pd.read_csv('/Users/zhj/Desktop/colony5.edges', header=None, sep="\s+")
ant_df.columns = (['source','target','weight','t'])


sources=np.unique(ant_df['source'])
targets=np.unique(ant_df['target'])
x=np.unique(ant_df['t'])
nodes=np.union1d(sources,targets)
T=len(x)
N=len(nodes)

dynamic_graphs=[]
dynamic_matrices=[]

for t in range(T):
    df=ant_df[ant_df['t']==x[t]][['source','target','weight']]
    edges=list(zip(df['source'],df['target'],df['weight']))

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges)
    G.remove_edges_from(nx.selfloop_edges(G))

    dynamic_graphs.append(G)

# Aggregated network
AGG=nx.Graph()
agg=np.zeros((N,N))
for go in dynamic_graphs:
    AGG=nx.compose(AGG,go)
    dynamic_matrices.append(nx.to_numpy_array(go))
    agg=agg+nx.to_numpy_array(go)

# Infomap
# dynamic_communities=[]
# for x in range(T):
#     G=dynamic_graphs[x]
#     # 使用Infomap算法进行社区分类
#     infomap = Infomap("--two-level --directed")
#     # 添加边到Infomap，包含权重
#     for u, v, data in G.edges(data=True):
#         weight=data['weight']
#         infomap.add_link(int(u), v, weight)
#
#     # 运行Infomap算法
#     infomap.run()
#
#     node_communities = np.zeros(N)
#     for node in infomap.tree:
#         if node.is_leaf:
#             node_communities[node.node_id-1]=node.module_id
#     dynamic_communities.append(node_communities)
#
#     # 如果不是第一个时间点，则与上一个时间点对齐
#     if x > 0:
#         node_communities = align_labels(dynamic_communities[-1], node_communities)
#
#     dynamic_communities.append(node_communities)

# Louvain
dynamic_communities = []
partition_list = []
for x in range(T):
    G = dynamic_graphs[x]
    partition = community_louvain.best_partition(G)
    partition_list.append(partition)

    node_communities = np.zeros(N, dtype=int)
    for node in partition:
        node_communities[node-1] = partition[node]

    # 如果不是第一个时间点，则与上一个时间点对齐
    if x > 0:
        node_communities = align_labels(dynamic_communities[-1], node_communities)

    dynamic_communities.append(node_communities)

new_dynamic_communities=[]
for x in range(T):
    node_communities=list(dynamic_communities[x])
    cnt=152
    for i in range(N):
        if len(np.where(node_communities==node_communities[i])[0])==1:
            node_communities[i]=155
            continue
        if node_communities.index(node_communities[i])==i:
            for j in range(i+1, N):
                if node_communities[j]==node_communities[i]:
                    node_communities[j]=cnt
            node_communities[i]=cnt
            cnt+=1
    node_communities=np.array(node_communities)
    node_communities-=152
    new_dynamic_communities.append(node_communities)


# dynamic_random=[]
# for t in range(T):
#     New_Graph = Random_Graph(dynamic_graphs[t],0.1)
#     new_graph = nx.to_numpy_array(New_Graph)
#     dynamic_random.append(new_graph)
static_pc=participation_coef(agg,new_dynamic_communities[0])
static_degrees=np.array([d for node,d in AGG.degree()])
pc=cal_pc(dynamic_matrices,new_dynamic_communities)
degrees=cal_degrees(dynamic_matrices)
dynamic_degrees=np.sum(degrees,axis=0)/T
dynamic_pc=np.sum(pc,axis=0)/T


# heavy_graphs, thin_graphs = filter(dynamic_graphs)
# dynamic_graphs=heavy_graphs
# dynamic_matrices=[]
# for G in dynamic_graphs:
#     dynamic_matrices.append(nx.to_numpy_array(G, weight='weight'))
#
# dynamic_random=[]
# for t in range(T):
#     New_Graph = Random_Graph(dynamic_graphs[t],0.1)
#     new_graph = nx.to_numpy_array(New_Graph)
#     dynamic_random.append(new_graph)
#
# dynamic_pc=cal_pc(dynamic_matrices,new_dynamic_communities)
# dynamic_degrees=cal_degrees(dynamic_matrices)
#
# pc=np.sum(dynamic_pc,axis=0)/len(dynamic_pc)
# res1=temporal_DC(dynamic_matrices,dynamic_random,pc,np.percentile(pc,80),1,T)
# degrees=np.sum(dynamic_degrees,axis=0)/len(dynamic_degrees)
# res2=temporal_RC(dynamic_matrices,dynamic_random,degrees,np.percentile(degrees,80),1,T)
# ant_df=pd.DataFrame({'TDC':res1,'TRC':res2})
# ant_df.to_csv('heavy0.25k0.8delta1.csv',index=False)
#
# # -----------------------------------------------
# dynamic_graphs=thin_graphs
# dynamic_matrices=[]
# for G in dynamic_graphs:
#     dynamic_matrices.append(nx.to_numpy_array(G, weight='weight'))
#
# dynamic_random=[]
# for t in range(T):
#     New_Graph = Random_Graph(dynamic_graphs[t],0.1)
#     new_graph = nx.to_numpy_array(New_Graph)
#     dynamic_random.append(new_graph)
#
# dynamic_pc=cal_pc(dynamic_matrices,new_dynamic_communities)
# dynamic_degrees=cal_degrees(dynamic_matrices)
#
# pc=np.sum(dynamic_pc,axis=0)/len(dynamic_pc)
# res1=temporal_DC(dynamic_matrices,dynamic_random,pc,np.percentile(pc,80),1,T)
# degrees=np.sum(dynamic_degrees,axis=0)/len(dynamic_degrees)
# res2=temporal_RC(dynamic_matrices,dynamic_random,degrees,np.percentile(degrees,80),1,T)
# ant_df=pd.DataFrame({'TDC':res1,'TRC':res2})
# ant_df.to_csv('thin0.25k0.8delta1.csv',index=False)