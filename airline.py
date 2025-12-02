import numpy as np
import networkx as nx
import pandas as pd
import random
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity

# static
def participation_coef(W, ci):
    '''
    Participation coefficient is a measure of diversity of intermodular
    connections of individual nodes.
    Parameters
    ----------
    W : NxN np.ndarray
        binary/weighted directed/undirected connection matrix
    ci : Nx1 np.ndarray
        community affiliation vector
    Returns
    -------
    P : Nx1 np.ndarray
        participation coefficient
    '''

    _, ci = np.unique(ci, return_inverse=True)
    ci += 1
    n = len(W)  # number of vertices
    Ko = np.sum(W, axis=1)  # (out) degree
    Gc = np.dot((W != 0), np.diag(ci))  # neighbor community affiliation, N*N
    Kc2 = np.zeros((n,))  # community-specific neighbors

    for i in range(1, int(np.max(ci)) + 1):
        Kc2 += np.square(np.sum(W * (Gc == i), axis=1))

    P = np.ones((n,)) - Kc2 / np.square(Ko)
    # P=0 if for nodes with no (out) neighbors
    P[np.where(np.logical_not(Ko))] = 0

    return P

# dynamic
def cal_pc(dy_mat, ci):
    dynamic_P=[]
    for t in range(T):
        W=dy_mat[t]
        n = len(W)
        Ko = np.sum(W, axis=1)
        Gc = np.dot((W != 0), np.diag(ci))
        Kc2 = np.zeros((n,))

        for i in range(1, int(np.max(ci)) + 1):
            Kc2 += np.square(np.sum(W * (Gc == i), axis=1))

        P = np.ones((n,)) - Kc2 / np.square(Ko)
        # P=0 if for nodes with no (out) neighbors
        P[np.where(np.logical_not(Ko))] = 0
        dynamic_P.append(P)
    dynamic_P = np.array(dynamic_P)
    return dynamic_P

# dynamic
def cal_degrees(dy_mat):
    dynamic_degrees=[]
    for t in range(T):
        W=dy_mat[t]
        degrees=np.sum(W,axis=1)
        dynamic_degrees.append(degrees)
    dynamic_degrees=np.array(dynamic_degrees)
    return dynamic_degrees

def inout_degrees(dy_mat, node_communities):
    time_steps = dy_mat.shape[0]
    num_nodes = dy_mat.shape[1]
    node_degrees = []

    for t in range(time_steps):
        in_degrees = np.zeros(num_nodes)
        out_degrees = np.zeros(num_nodes)

        for i in range(num_nodes):
            community_i = node_communities[i]

            for j in range(num_nodes):
                if dy_mat[t][i][j] > 0:
                    community_j = node_communities[j]
                    if community_i == community_j:
                        in_degrees[i] += 1
                    else:
                        out_degrees[i] += 1

        node_degrees.append((in_degrees, out_degrees))

    return node_degrees

def Random_Graph(G, p):
    original_nodes = list(G.nodes())
    original_edges = list(G.edges())
    num_edges = len(original_edges)
    degrees = dict(G.degree())

    new_graph = nx.Graph()
    new_graph.add_nodes_from(original_nodes)
    new_graph.add_edges_from(original_edges)
    edges_to_remove = random.sample(original_edges, int(num_edges * p))

    for edge in edges_to_remove:
        new_graph.remove_edge(edge[0], edge[1])

    # randomly place back in the graph
    for _ in range(len(edges_to_remove)):
        current_degrees = dict(new_graph.degree())
        for _ in range(N):
            possible_nodes = [node for node in G.nodes if current_degrees[node] < degrees[node]]
            if len(possible_nodes) < 2:
                break
            n1, n2 = random.sample(possible_nodes, 2)

            # check if edge already exists
            if not new_graph.has_edge(n1, n2):
                new_graph.add_edge(n1, n2)
                break
    return new_graph


def clubness(g, random_g, index_k):
    # the ratio of original graph and random graph
    size_k=len(index_k)
    sum_g=0
    sum_random=0
    for i in index_k:
        for j in index_k:
            sum_g+=g[i][j]
            sum_random+=random_g[i][j]
    return sum_g/(size_k*(size_k-1))

def temporal_DC(dy_mat, dy_ran, pc, k, delta, T):
    '''
    dy_mat and dy_ran: TxNxN np.ndarray
    pc: Nx1 np.array
    k and delta: hyerparameter
    return c: a list of clubness from 0 to T-delta-1
    '''
    index_k=np.array(np.where(pc >= k)[0])
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
        c[t]=clubness(a, b, index_k)
    return c

def temporal_RC(dy_mat, dy_ran, degrees, k, delta, T):
    index_k = np.array(np.where(degrees >= k)[0])
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

def temporal_DRC(dy_mat, dy_ran, pc, degrees, k_DC, k_RC, delta, T):
    index_DC = np.array(np.where(pc >= k_DC)[0])
    index_RC = np.array(np.where(degrees >= k_RC)[0])
    index_k = set(index_DC) & set(index_RC)
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

def Efficiency(dy_gra,node_class=np.arange(1920)):
    efficiency_list = []
    for G in dy_gra:
        G=G.subgraph(node_class)
        eff = nx.global_efficiency(G)
        efficiency_list.append(eff)
    return efficiency_list

def Modularity(dy_gra,node_class=np.arange(1920)):
    modularity_list = []
    for G in dy_gra:
        if G.number_of_edges() == 0:
            modularity_list.append(0.0)
            continue
        G=G.subgraph(node_class)
        communities = list(greedy_modularity_communities(G))
        mod = modularity(G, communities)
        modularity_list.append(mod)
    return modularity_list

def clustering_coefficient(dy_gra,node_class=np.arange(1920)):
    clustering_list=[]
    for G in dy_gra:
        G=G.subgraph(node_class)
        clustering_list.append(nx.average_clustering(G))
    return clustering_list

def kuramoto_model(dy_mat, K=1.0, dt=0.01, node_class=np.arange(1920), omega=None, theta0=None, seed=None):
    """
    Simulate Kuramoto Model

    Parameters：
    - K: coupling strength
    - dt: time step
    - omega: natural frequency array (optional)
    - theta0: initial phase array (optional)

    Return：
    - theta_history: shape (T, N)
    - R_history: shape (T,)
    """

    if seed is not None:
        np.random.seed(seed)

    if omega is None:
        omega = np.random.normal(loc=0.0, scale=0.1, size=N)
    if theta0 is None:
        theta0 = np.random.uniform(0, 2 * np.pi, size=N)

    theta = theta0.copy()
    theta_list = np.zeros((T, N))
    R_list = np.zeros(T)

    for t in range(T):
        dtheta = np.zeros((N,))
        A=dy_mat[t]
        for i in node_class:
            interaction = 0
            for j in node_class:
                if A[i, j] != 0:
                    interaction += np.sin(theta[j] - theta[i])
            dtheta[i] = omega[i] + K * interaction

        theta += dt * dtheta
        theta_list[t] = theta

        # global R(t)
        r_complex = np.exp(1j * theta)
        R_list[t] = np.abs(np.sum(r_complex[node_class]) / len(node_class))

    return theta_list, R_list

def save_matrices(data, save_path):
    np.savez_compressed(save_path, dynamic_matrices=np.array(data))

def load_matrices(load_path):
    data = np.load(load_path, allow_pickle=False)
    return data["dynamic_matrices"]

USAL_TN_df = pd.read_csv('/home/zhuanghaojun/airline/USAL_TN_monthly_2012_2020.txt', header=None, sep="\s+")
USAL_TN_df.columns = (['t','i','j'])
ordered_IDs = pd.read_csv('/home/zhuanghaojun/airline/airport.txt', header=None, sep="\s+")
ordered_IDs.columns = (['index','name','community'])

iis=np.unique(USAL_TN_df['i'])
jjs=np.unique(USAL_TN_df['j'])
nodes=np.union1d(iis,jjs)
x=np.unique(USAL_TN_df['t'])
N=len(nodes)
T=len(x)
dynamic_graphs=[]
dynamic_matrices=[]

# Temporal networks
for t in range(T):
    fr=USAL_TN_df[USAL_TN_df['t']==x[t]][['i','j']]
    gu=nx.from_pandas_edgelist(fr,'i','j')
    g=nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(gu.edges)
    g.remove_edges_from(nx.selfloop_edges(g))
    dynamic_graphs.append(g)

# Aggregated network
AGG=nx.Graph()
agg=np.zeros((N,N))
for go in dynamic_graphs:
    AGG=nx.compose(AGG,go)
    dynamic_matrices.append(nx.to_numpy_array(go))
    agg=agg+nx.to_numpy_array(go)


# four communities
# area1={'CT','MA','ME','NH','NJ','NY','PA','RI','VT','DE','MD'}
# area2={'ND','SD','NE','KS','MN','IA','MO','WI','IL','MI','IN','OH'}
# area3={'OK','TX','AR','LA','KY','TN','MS','AL','WV','VA','GA','NC','SC','FL','DC','VI','PR','TT'}
# area4={'AK','HI','WA','OR','CA','ID','NV','UT','AZ','MT','WY','CO','NM'}
#
# node_communities=np.zeros((N,))
# for i in range(N):
#     if ordered_IDs['community'][i] in area1:
#         node_communities[i]=1
#     if ordered_IDs['community'][i] in area2:
#         node_communities[i]=2
#     if ordered_IDs['community'][i] in area3:
#         node_communities[i]=3
#     if ordered_IDs['community'][i] in area4:
#         node_communities[i]=4
# node_communities=node_communities.astype(int)

# nine communities
area1={'ME','VT','NH','MA','CT','RI'}
area2={'PA','NY','NJ','DE','MD'}
area3={'ND','SD','NE','KS','MN','IA','MO'}
area4={'WI','IL','MI','IN','OH'}
area5={'OK','TX','AR','LA'}
area6={'KY','TN','MS','AL'}
area7={'WV','VA','GA','NC','SC','FL','DC','VI','PR','TT'}
area8={'AK','HI','WA','OR','CA'}
area9={'ID','NV','UT','AZ','MT','WY','CO','NM'}

node_communities=np.zeros((N,))
for i in range(N):
    if ordered_IDs['community'][i] in area1:
        node_communities[i]=1
    if ordered_IDs['community'][i] in area2:
        node_communities[i]=2
    if ordered_IDs['community'][i] in area3:
        node_communities[i]=3
    if ordered_IDs['community'][i] in area4:
        node_communities[i]=4
    if ordered_IDs['community'][i] in area5:
        node_communities[i]=5
    if ordered_IDs['community'][i] in area6:
        node_communities[i]=6
    if ordered_IDs['community'][i] in area7:
        node_communities[i]=7
    if ordered_IDs['community'][i] in area8:
        node_communities[i]=8
    if ordered_IDs['community'][i] in area9:
        node_communities[i]=9
node_communities=node_communities.astype(int)

# dynamic_random=[]
# p = 0.2
# for t in range(T):
#     New_Graph = Random_Graph(dynamic_graphs[t],p)
#     new_graph = nx.to_numpy_array(New_Graph)
#     dynamic_random.append(new_graph)
static_pc=participation_coef(agg,node_communities)
static_degrees=np.array([d for node,d in AGG.degree()])
pc=cal_pc(dynamic_matrices,node_communities)
degrees=cal_degrees(dynamic_matrices)
dynamic_degrees=np.sum(degrees,axis=0)/T
dynamic_pc=np.sum(pc,axis=0)/T

save_matrices(dynamic_pc,save_path="dynamic_pc.npz")
save_matrices(dynamic_degrees,save_path="dynamic_degrees.npz")

# index=[i for i in range(1920)]
# filename = f"dynamic_pc.csv"
# with open(filename, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(index)
#     for t in range(T):
#         writer.writerow(pc[t])
#
# index=[i for i in range(1920)]
# filename = f"dynamic_degrees.csv"
# with open(filename, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(index)
#     for t in range(T):
#         writer.writerow(degrees[t])

# node_degrees=inout_degrees(dynamic_matrices,node_communities)
# in_degrees_all = []
# out_degrees_all = []
# for _, (in_degrees, out_degrees) in enumerate(node_degrees):
#     in_degrees_all.append(in_degrees.tolist())
#     out_degrees_all.append(out_degrees.tolist())


# ------------clustering coefficient
# r2020_df = pd.read_excel('/Users/zhj/Desktop/2020.xlsx', header=None)
# r2020_df.columns = (['Ranking','Name','Id'])
# id_2020=np.array(r2020_df['Id'])
#
# pc_ranking=sorted(id_2020, key= lambda i:pc[i],reverse=True)
# degrees_ranking=sorted(id_2020, key= lambda i:degrees[i],reverse=True)

# TDC_dynamic=clustering_coefficient(dynamic_graphs,index_TDC)
# TDC_static=clustering_coefficient(dynamic_graphs,index_SDC)
# TRC_dynamic=clustering_coefficient(dynamic_graphs,index_TRC)
#
# ce_df=pd.DataFrame({'TDC_dynamic':TDC_dynamic,'TDC_static':TDC_static,'TRC_dynamic':TRC_dynamic})
# ce_df.to_csv('ce80.csv',index=False)

# ------------modularity and efficiency
# TDC_dynamic=Modularity(dynamic_graphs,index_TDC)
# TDC_static=Modularity(dynamic_graphs,index_SDC)
# TRC_dynamic=Modularity(dynamic_graphs,index_TRC)
#
# ce_df=pd.DataFrame({'TDC_dynamic':TDC_dynamic,'TDC_static':TDC_static,'TRC_dynamic':TRC_dynamic})
# ce_df.to_csv('mod80.csv',index=False)

# ------------kuramoto model
# _, TDC_dynamic=kuramoto_model(dynamic_matrices,node_class=index_TDC)
# _, TDC_static=kuramoto_model(dynamic_matrices,node_class=index_SDC)
# _, TRC_dynamic=kuramoto_model(dynamic_matrices,node_class=index_TRC)
# kuramoto_df=pd.DataFrame({'TDC_dynamic':TDC_dynamic,'TDC_static':TDC_static,'TRC':TRC_dynamic})
# kuramoto_df.to_csv('kuramoto80.csv',index=False)

# ------------Supplementary
from itertools import combinations
from collections import defaultdict
def k_cliques(adj_matrix, node_class, k):
    """
    strategy: find Maximal Cliques then discompose to the order scale
    faster than enumerate_all_cliques
    """
    sub_adj = adj_matrix[np.ix_(node_class, node_class)]
    G = nx.from_numpy_array(sub_adj)

    cliques = set()

    for max_c in nx.find_cliques(G):
        len_c = len(max_c)
        if len_c < k:
            continue

        global_nodes_sorted = sorted([node_class[i] for i in max_c])
        for subset in combinations(global_nodes_sorted, k):
            cliques.add(subset)

    return cliques


def longest_lifespan(times):
    ts = sorted(times)
    max_len = cur_len = 1
    for i in range(1, len(ts)):
        if ts[i] == ts[i - 1] + 1:
            cur_len += 1
        else:
            cur_len = 1
        max_len = max(max_len, cur_len)
    return max_len

def evaluate_model(dy_mat, node_class, order):
    T = len(dy_mat)
    lifespans = defaultdict(list)

    for t in range(T):
        cliques_t = k_cliques(dy_mat[t], node_class, order)
        for clique in cliques_t:
            lifespans[clique].append(t)

    # ---statistic
    count = len(lifespans)

    if count > 0:
        all_lifespans = [longest_lifespan(times) for times in lifespans.values()]
        mean_l = np.mean(all_lifespans)
        std_l = np.std(all_lifespans) if len(all_lifespans) > 1 else 0.0
    else:
        mean_l, std_l = 0.0, 0.0

    return {
        'count': count,
        'mean_lifespan': mean_l,
        'std_lifespan': std_l
    }