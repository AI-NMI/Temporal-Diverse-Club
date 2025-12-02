import os
import h5py
import random
import numpy as np
import networkx as nx
from sklearn.preprocessing import MinMaxScaler

def participation_coef(W, ci):
    '''
    Participation coefficient is a measure of diversity of intermodular
    connections of individual nodes.
    Parameters
    ----------
    W : NxN np.ndarray
        binary/weighted directed/undirected connection matrix
    ci : Nx1 np.ndarray
        community affiliation vector社区从属矩阵

    Returns
    -------
    P : Nx1 np.ndarray
        participation coefficient
    '''

    _, ci = np.unique(ci, return_inverse=True)
    ci += 1
    n = len(W)  # number of vertices
    Ko = np.sum(W, axis=1)  # (out) degree加权出度
    Gc = np.dot((W != 0), np.diag(ci))  # neighbor community affiliation, N*N
    Kc2 = np.zeros((n,))  # community-specific neighbors

    for i in range(1, int(np.max(ci)) + 1):
        Kc2 += np.square(np.sum(W * (Gc == i), axis=1))

    P = np.ones((n,)) - Kc2 / np.square(Ko)
    # P=0 if for nodes with no (out) neighbors
    P[np.where(np.logical_not(Ko))] = 0

    return P

def Random_Graph(G, p):
    original_nodes = list(G.nodes())
    original_edges = list(G.edges())
    num_edges = len(original_edges)
    num_nodes = len(original_nodes)
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
        for _ in range(num_nodes):
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
    sum_g=0
    sum_random=0
    for i in index_k:
        for j in index_k:
            sum_g += g[i][j]
            sum_random += random_g[i][j]
    if sum_random==0: return 1
    return sum_g/sum_random

def temporal_DC(dy_mat, dy_ran, pc, k, delta, T):
    '''
    dy_mat and dy_ran: TxNxN np.ndarray
    pc: Nx1 np.array
    k and delta: hyperparameter
    return c: a list of clubness from 0 to T-delta-1
    '''
    index_k=np.array(np.where(pc >= k)[0])
    size_k=len(index_k)
    c=np.zeros((T - delta,))
    for t in range(T - delta):
        a=np.array(dy_mat[t])
        b=np.array(dy_ran[t])
        for d in range(delta+1):
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
        for d in range(delta+1):
            for i in index_k:
                for j in index_k:
                    if dy_mat[t + d][i][j] == 0:
                        a[i][j] = 0
                    if dy_ran[t + d][i][j] == 0:
                        b[i][j] = 0
        c[t] = clubness(a, b, index_k)
    return c

def temporal_DRC(dy_mat, dy_ran, pc, degrees, k_DC, k_RC, delta, T):
    index_DC=np.array(np.where(pc >= k_DC)[0])
    index_RC=np.array(np.where(degrees >= k_RC)[0])
    index_k=set(index_DC)&set(index_RC)
    size_k = len(index_k)
    c = np.zeros((T - delta,))
    for t in range(T - delta):
        a = np.array(dy_mat[t])
        b = np.array(dy_ran[t])
        for d in range(delta+1):
            for i in index_k:
                for j in index_k:
                    if dy_mat[t + d][i][j] == 0:
                        a[i][j] = 0
                    if dy_ran[t + d][i][j] == 0:
                        b[i][j] = 0
        c[t] = clubness(a, b, index_k)
    return c


def load_Control():
    directory = '/Users/zhj/Desktop/Control_SP_BN246'
    mat_Control = []
    for filename in os.listdir(directory):
        if filename.endswith(".mat"):
            filepath = os.path.join(directory, filename)
            with h5py.File(filepath, 'r') as f:
                file_data = [np.array(f[key]) for key in f.keys()]
            mat_Control.append(file_data[0])
    return mat_Control

def load_Patient():
    directory = '/Users/zhj/Desktop/Patient_SP_BN246'
    mat_Patient = []
    for filename in os.listdir(directory):
        if filename.endswith(".mat"):
            filepath = os.path.join(directory, filename)
            with h5py.File(filepath, 'r') as f:
                file_data = [np.array(f[key]) for key in f.keys()]
            mat_Patient.append(file_data[0])
    return mat_Patient


def spar(normalized_matrix, in_community_sparseness, out_community_sparseness):
    flatten = normalized_matrix.flatten()
    in_community_threshold = np.percentile(flatten, in_community_sparseness)
    out_community_threshold = np.percentile(flatten, out_community_sparseness)
    for i, j in in_community:
        if normalized_matrix[i][j] >= in_community_threshold:
            normalized_matrix[i][j] = 1
        else:
            normalized_matrix[i][j] = 0
    for i, j in out_community:
        if normalized_matrix[i][j] >= out_community_threshold:
            normalized_matrix[i][j] = 1
        else:
            normalized_matrix[i][j] = 0
    return normalized_matrix

def init_Control(mat_data,n):
    global dynamic_matrices_Control
    global dynamic_graphs_Control

    roi_timeseries = mat_data[n]
    roi_timeseries = roi_timeseries.T

    window_size = 20
    step_size = 1
    num_rois = roi_timeseries.shape[1]
    num_windows = (roi_timeseries.shape[0] - window_size) // step_size + 1

    for start in range(0, roi_timeseries.shape[0] - window_size + 1, step_size):
        window_data = roi_timeseries[start:start + window_size, :]  # BOLD signal during the window
        fc_matrix = np.corrcoef(window_data.T)  # Pearson Correlation
        normalized_matrix = MinMaxScaler().fit_transform(fc_matrix)
        normalized_matrix = spar(normalized_matrix,50,50) # normalize edges to 0/1
        normalized_matrix[range(normalized_matrix.shape[0]), range(normalized_matrix.shape[1])] = 0
        dynamic_matrices_Control.append(normalized_matrix)

        G = nx.Graph()
        nodes=np.array(range(normalized_matrix.shape[0]))
        G.add_nodes_from(list(nodes))
        for i, j in [(i, j) for i in range(normalized_matrix.shape[0]) for j in range(normalized_matrix.shape[1])]:
            if normalized_matrix[i][j] != 0:
                G.add_edge(i, j)
        G.remove_edges_from(nx.selfloop_edges(G))
        dynamic_graphs_Control.append(G)

    T = len(dynamic_matrices_Control)
    # Aggregated network
    global agg_Control
    global AGG_Control
    for t in range(T):
        agg_Control += dynamic_matrices_Control[t]
        AGG_Control = nx.compose(AGG_Control,dynamic_graphs_Control[t])
    return

def init_Patient(mat_data,n):
    global dynamic_matrices_Patient
    global dynamic_graphs_Patient

    roi_timeseries = mat_data[n]
    roi_timeseries = roi_timeseries.T

    window_size = 20
    step_size = 1
    num_rois = roi_timeseries.shape[1]
    num_windows = (roi_timeseries.shape[0] - window_size) // step_size + 1

    for start in range(0, roi_timeseries.shape[0] - window_size + 1, step_size):
        window_data = roi_timeseries[start:start + window_size, :]
        fc_matrix = np.corrcoef(window_data.T)
        normalized_matrix = MinMaxScaler().fit_transform(fc_matrix)
        normalized_matrix = spar(normalized_matrix,50,50)
        normalized_matrix[range(normalized_matrix.shape[0]), range(normalized_matrix.shape[1])] = 0
        dynamic_matrices_Patient.append(normalized_matrix)

        G = nx.Graph()
        nodes = np.array(range(normalized_matrix.shape[0]))
        G.add_nodes_from(list(nodes))
        for i, j in [(i, j) for i in range(normalized_matrix.shape[0]) for j in range(normalized_matrix.shape[1])]:
            if normalized_matrix[i][j] != 0:
                G.add_edge(i, j)
        G.remove_edges_from(nx.selfloop_edges(G))
        dynamic_graphs_Patient.append(G)

    T = len(dynamic_matrices_Patient)
    # Aggregated network
    global agg_Patient
    global AGG_Patient
    for t in range(T):
        agg_Patient += dynamic_matrices_Patient[t]
        AGG_Patient = nx.compose(AGG_Patient, dynamic_graphs_Patient[t])
    return


#sample
mat_Control=load_Control()
mat_Patient=load_Patient()

# 160
# label1=np.array([1,2,2,1,1,1,1,3,2,2,1,2,1,1,1,2,1,3,3,1,2,2,2,2,3,3,3,3,2,3,3,4,3,2,4,2,4,3,3,3])
# label2=np.array([4,4,4,3,4,4,3,4,4,4,4,4,4,4,4,4,3,3,3,4,3,4,1,4,4,4,4,4,4,4,3,1,1,4,4,3,4,3,4,3])
# label3=np.array([3,4,4,1,1,4,3,2,3,1,1,1,1,1,3,2,3,5,2,3,2,3,3,2,1,6,2,1,5,5,1,1,5,2,1,2,1,6,6,5])
# label4=np.array([5,5,6,1,3,6,5,5,6,5,5,1,6,1,6,1,1,5,6,5,1,6,5,5,6,1,6,6,6,5,5,6,6,6,5,6,6,6,6,6])
# node_communities=np.concatenate((label1,label2,label3,label4))

# 246
node_communities=np.zeros((246,),dtype=int)
for i in range(246):
    if i>=0 and i<=67: node_communities[i]=1
    elif i>=68 and i<=123: node_communities[i]=2
    elif i>=124 and i<=161: node_communities[i]=3
    elif i>=162 and i<=173: node_communities[i]=4
    elif i>=174 and i<=187: node_communities[i]=5
    elif i>=188 and i<=209: node_communities[i]=6
    else: node_communities[i]=7

# 400
# node_communities=np.zeros((400,),dtype=int)
# for i in range(400):
#     if i<=30: node_communities[i]=1
#     elif i>=31 and i<=67: node_communities[i]=2
#     elif i>=68 and i<=90: node_communities[i]=3
#     elif i>=91 and i<=112: node_communities[i]=4
#     elif i>=113 and i<=125: node_communities[i]=5
#     elif i>=126 and i<=147: node_communities[i]=6
#     elif i>=148 and i<=199: node_communities[i]=7
#
#     elif i>=200 and i<=229: node_communities[i]=1
#     elif i>=230 and i<=269: node_communities[i]=2
#     elif i>=270 and i<=292: node_communities[i]=3
#     elif i>=293 and i<=317: node_communities[i]=4
#     elif i>=318 and i<=330: node_communities[i]=5
#     elif i>=331 and i<=360: node_communities[i]=6
#     else : node_communities[i]=7

in_community=[]
out_community=[]
for i in range(len(node_communities)):
    for j in range(len(node_communities)):
        if node_communities[i] == node_communities[j]:
            in_community.append((i, j))
        else:
            out_community.append((i, j))

X_TDC=[]
X_TRC=[]
X_DRC=[]
y=[]
for i in range(len(mat_Control)):
    dynamic_matrices_Control = []
    dynamic_graphs_Control = []
    agg_Control = np.zeros((246,246))
    AGG_Control = nx.Graph()
    init_Control(mat_Control,i)
    dynamic_random_Control=[]
    T = len(dynamic_matrices_Control)
    N = len(dynamic_matrices_Control[0])
    for t in range(T):
        New_Graph = Random_Graph(dynamic_graphs_Control[t],0.2)
        new_graph = nx.to_numpy_array(New_Graph)
        dynamic_random_Control.append(new_graph)
    pc_Control = participation_coef(agg_Control, node_communities)
    degrees_Control=np.sum(agg_Control,axis=0)
    # pc=cal_pc(dynamic_matrices_Control,node_communities)
    # degrees=cal_degrees(dynamic_matrices_Control)
    sub_TDC=[]
    sub_TRC=[]
    sub_DRC=[]
    for k in [50,60,70,80,90]:
        for delta in range(5):
            TDC_Control = temporal_DC(dynamic_matrices_Control, dynamic_random_Control, pc_Control,
                                      np.percentile(pc_Control, k), delta, T)
            TRC_Control = temporal_RC(dynamic_matrices_Control, dynamic_random_Control, degrees_Control,
                                      np.percentile(degrees_Control, k), delta, T)
            DRC_Control = temporal_DRC(dynamic_matrices_Control, dynamic_random_Control, pc_Control, degrees_Control,
                                       np.percentile(pc_Control, k), np.percentile(degrees_Control, k), delta, T)
            sub_TDC.append(TDC_Control)
            sub_TRC.append(TRC_Control)
            sub_DRC.append(DRC_Control)

    X_TDC.append(sub_TDC)
    X_TRC.append(sub_TRC)
    X_DRC.append(sub_DRC)
    y.append(0)

for i in range(len(mat_Patient)):
    dynamic_matrices_Patient = []
    dynamic_graphs_Patient = []
    agg_Patient = np.zeros((246,246))
    AGG_Patient = nx.Graph()
    init_Patient(mat_Patient,i)
    dynamic_random_Patient=[]
    T = len(dynamic_matrices_Patient)
    N = len(dynamic_matrices_Patient[0])
    for t in range(T):
        New_Graph = Random_Graph(dynamic_graphs_Patient[t],0.2)
        new_graph = nx.to_numpy_array(New_Graph)
        dynamic_random_Patient.append(new_graph)
    pc_Patient = participation_coef(agg_Patient, node_communities)
    degrees_Patient = np.sum(agg_Patient,axis=0)
    # pc=cal_pc(dynamic_matrices_Patient,node_communities)
    # degrees=cal_degrees(dynamic_matrices_Patient)
    sub_TDC=[]
    sub_TRC=[]
    sub_DRC=[]
    for k in [50,60,70,80,90]:
        for delta in range(5):
            TDC_Patient = temporal_DC(dynamic_matrices_Patient, dynamic_random_Patient, pc_Patient,
                                      np.percentile(pc_Patient, k), delta, T)
            TRC_Patient = temporal_RC(dynamic_matrices_Patient, dynamic_random_Patient, degrees_Patient,
                                      np.percentile(degrees_Patient, k), delta, T)
            DRC_Patient = temporal_DRC(dynamic_matrices_Patient, dynamic_random_Patient, pc_Patient, degrees_Patient,
                                       np.percentile(pc_Patient, k), np.percentile(degrees_Patient, k), delta, T)
            sub_TDC.append(TDC_Patient)
            sub_TRC.append(TRC_Patient)
            sub_DRC.append(DRC_Patient)

    X_TDC.append(sub_TDC)
    X_TRC.append(sub_TRC)
    X_DRC.append(sub_DRC)
    y.append(1)

y=np.array(y)

# ------------ clubness for machine learning (traditional)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# =====================
# Step 1: preprocessing
# =====================
def aggregate_features(X_subjects, stats=None):
    """
    Input:
        X_subjects: np.ndarray, shape (K * Δ, T) 单个被试的数据
        stats: statistic metrics
        Attention: here T is changeable!!!
    Output:
        feature_vector: np.ndarray, shape (K * Δ * len(stats),)
    """
    feats = []

    for ts in X_subjects:
        ts = np.array(ts, dtype=float)
        vals = []
        if "mean" in stats:
            vals.append(np.mean(ts))
        if "std" in stats:
            vals.append(np.std(ts))
        if "min" in stats:
            vals.append(np.min(ts))
        if "max" in stats:
            vals.append(np.max(ts))
        feats.extend(vals)

    return np.array(feats)


def build_feature_matrix(X_all, stats=["mean", "std", "min", "max"]):

    return np.stack([aggregate_features(x, stats=stats) for x in X_all])


# =====================
# Step 2: evaluation
# =====================

param_grid = {
    "LogReg": {
        "clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "clf__penalty": ["l1", "l2"],
        "clf__solver": ["liblinear", "lbfgs"]
    },
    "SVM": {
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__kernel": ["linear", "rbf"],
        "clf__gamma": [0.001, 0.01, 0.1, 1, 10]
    },
    "RF": {
        "clf__n_estimators": [100, 300, 500, 1000],
        "clf__max_depth": [None, 5, 10, 20, 50],
        "clf__max_features": ["sqrt", "log2", None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4]
    },
    "KNN": {
        "clf__n_neighbors": [3, 5, 7, 9, 15],
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2],
        "clf__metric": ["minkowski", "chebyshev"]
    }
}

models = {
    "LogReg": LogisticRegression(max_iter=500),
    "SVM": SVC(probability=True),
    "RF": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier()
}


def evaluate_models(X, y, label, n_splits=5):
    results = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for model_name, model in models.items():
        accs, f1s, recalls, precisions, specificities, aucs = [], [], [], [], [], []

        # 外层交叉验证
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # pipeline
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", model)
            ])

            # 内层 GridSearchCV 调参
            grid = GridSearchCV(
                pipeline,
                param_grid[model_name],
                cv=3,
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_

            y_pred = best_model.predict(X_test)

            accs.append(accuracy_score(y_test, y_pred))
            f1s.append(f1_score(y_test, y_pred, zero_division=0))
            recalls.append(recall_score(y_test, y_pred, zero_division=0))
            precisions.append(precision_score(y_test, y_pred, zero_division=0))

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specificities.append(specificity)

        results.append({
            "FeatureSet": label,
            "Model": model_name,
            "Accuracy_mean": np.mean(accs),
            "Accuracy_std": np.std(accs),
            "F1_mean": np.mean(f1s),
            "F1_std": np.std(f1s),
            "Recall_mean": np.mean(recalls),
            "Recall_std": np.std(recalls),
            "Precision_mean": np.mean(precisions),
            "Precision_std": np.std(precisions),
            "Specificity_mean": np.mean(specificities),
            "Specificity_std": np.std(specificities),

        })

    return results


# =====================
# Step 3: main
# =====================

# X_TDC = build_feature_matrix(X_TDC)
# X_TRC = build_feature_matrix(X_TRC)
# X_DRC = build_feature_matrix(X_DRC)
#
# results_div = evaluate_models(X_TDC, y, label="TDC")
# results_rich = evaluate_models(X_TRC, y, label="TRC")
# results_div_rich = evaluate_models(X_DRC, y, label="DRC")
#
# df_results = pd.DataFrame(results_div + results_rich + results_div_rich)
# df_results.to_csv("club_features.csv", index=False)

# ------------ clubness for machine learning (2D-CNN)
def padding(X_subjects, max_len=None, pad_value=0):
    """
    Every subject has the same shape!!!
    data_list: list of (25, T_i)
    man_len: when delta=1
    """
    if max_len is None:
        max_len = max(len(x[0]) for x in X_subjects)
    padded_list = []
    for x in X_subjects:
        subject_padded = []
        for row in x:
            T=len(row)
            if T < max_len:
                x_pad = list(row) + [pad_value] * (max_len-T)
            else:
                x_pad = row[:max_len]
            subject_padded.append(x_pad)
        padded_list.append(np.array(subject_padded,dtype=np.float32))
    return np.stack(padded_list)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        # data augment
        if self.augment and np.random.rand() < 0.3:
            x += np.random.normal(0, 0.05, size=x.shape).astype(np.float32)
        x = np.expand_dims(x, 0)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

class CNN2D(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,kernel_size=(3,3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128,kernel_size=(3,3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ---------------- K-Fold CV ----------------
def evaluate_model(X, y, label="", K=5, epochs=30, batch_size=16, lr=1e-3):
    from sklearn.model_selection import KFold

    results = []
    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    accs, f1s, recalls, precisions, specificities, aucs = [], [], [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        train_set = Subset(TimeSeriesDataset(X, y, augment=True), train_idx)
        val_set   = Subset(TimeSeriesDataset(X, y, augment=False), val_idx)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        model = CNN2D(T=X.shape[2]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # train
        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        # test
        model.eval()
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                prob = F.softmax(logits, dim=1)[:,1]
                pred = logits.argmax(dim=1)
                y_true.extend(yb.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
                y_prob.extend(prob.cpu().numpy())

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp + 1e-6)

        accs.append(acc)
        f1s.append(f1)
        recalls.append(recall)
        precisions.append(precision)
        specificities.append(specificity)
        aucs.append(auc)

    results.append({
        "Label": label,
        "Accuracy_mean": np.mean(accs), "Accuracy_std": np.std(accs),
        "F1_mean": np.mean(f1s), "F1_std": np.std(f1s),
        "Recall_mean": np.mean(recalls), "Recall_std": np.std(recalls),
        "Precision_mean": np.mean(precisions), "Precision_std": np.std(precisions),
        "Specificity_mean": np.mean(specificities), "Specificity_std": np.std(specificities),
        "AUC_mean": np.mean(aucs), "AUC_std": np.std(aucs)
    })

    return results

# X_TDC=padding(X_TDC)
# X_TRC=padding(X_TRC)
# X_DRC=padding(X_DRC)
#
# results_div = evaluate_model(X_TDC, y, label="TDC")
# results_rich = evaluate_model(X_TRC, y, label="TRC")
# results_div_rich = evaluate_model(X_DRC, y, label="DRC")
#
# df_results = pd.DataFrame(results_div + results_rich + results_div_rich)
# df_results.to_csv("club_features.csv", index=False)