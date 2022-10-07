import numpy as np
import torch
import torch.utils.data as Data
import os
from utils.logs import log_string
import pandas as pd


def max_min_normalization(x, _max, _min):
    x = (x - _max) / _min
    return x


def generate_seq(data, train_length, pred_length):
    seq = np.concatenate([np.expand_dims(
        data[i: i + train_length + pred_length], 0)
        for i in range(data.shape[0] - train_length - pred_length + 1)],
        axis=0)[:, :, :, 0: 1]
    return np.split(seq, 2, axis=1)


def generate_from_data(data, length, missing_rate, transformer):
    mean = None
    std = None
    train_line, val_line = int(length * 0.6), int(length * 0.8)
    for line1, line2 in ((0, train_line),
                         (train_line, val_line),
                         (val_line, length)):
        x, y = generate_seq(data['data'][line1: line2], 12, 12)



        if transformer:
            x = transformer(x)
            y = transformer(y)
        if mean is None:
            mean = x.mean()
        if std is None:
            std = x.std()

        yield (x - mean) / std, y


def generate_from_train_val_test(data, transformer):
    mean = None
    std = None
    for key in ('train', 'val', 'test'):
        x, y = generate_seq(data[key], 12, 12)

        if transformer:
            x = transformer(x)
            y = transformer(y)
        if mean is None:
            mean = x.mean()
        if std is None:
            std = x.std()

        yield (x - mean) / std, y


def generate_data(graph_signal_matrix_filename, missing_rate, transformer=None):
    '''
    shape is (num_of_samples, 12, num_of_vertices, 1)
    '''
    data = np.load(graph_signal_matrix_filename)
    keys = data.keys()
    if 'train' in keys and 'val' in keys and 'test' in keys:
        for i in generate_from_train_val_test(data, transformer):
            yield i
    elif 'data' in keys:
        length = data['data'].shape[0]
        for i in generate_from_data(data, length, missing_rate, transformer):
            yield i
    else:
        raise KeyError("neither data nor train, val, test is in the data")


def load_data(graph_signal_matrix_filename, batch_size, test, log, missing_rate):
    loaders = []
    true_values = []

    last_length = 0
    for idx, (x, y) in enumerate(generate_data(graph_signal_matrix_filename, missing_rate)):
        if test:
            x = x[: 100]
            y = y[: 100]


        y = y.squeeze(axis=-1)
        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)

        data = Data.TensorDataset(x, y)

        loaders.append(
            Data.DataLoader(dataset=data,
                            batch_size=batch_size,
                            # pin_memory=True,
                            # num_workers=0,
                            drop_last=False,  # (idx == 0),
                            shuffle=(idx == 0),
                            )
        )

        if idx == 0:
            training_samples = x.shape[0]
            log_string(log, 'training')
            log_string(log, 'input shape:{}'.format(x.shape))
            log_string(log, 'output shape:{}\n'.format(y.shape))
        if idx == 1:
            val_samples = x.shape[0]
            log_string(log, 'validation')
            log_string(log, 'input shape:{}'.format(x.shape))
            log_string(log, 'output shape:{}\n'.format(y.shape))
        if idx == 2:
            test_sample = x.shape[0]
            log_string(log, 'testing')
            log_string(log, 'input shape:{}'.format(x.shape))
            log_string(log, 'output shape:{}\n'.format(y.shape))

    train_loader, val_loader, test_loader = loaders

    return train_loader, val_loader, test_loader, training_samples, val_samples, test_sample


def load_adj(adj_dtw_filename, graph_signal_matrix_filename, adj_filename, num_of_vertices, id_filename, model_name, log):
    if id_filename is not None:
        if not os.path.exists(id_filename):
            id_filename = None

    train_data = np.load(graph_signal_matrix_filename)
    train_data = train_data['data']
    train_data = train_data[:int(0.6 * len(train_data)), :, 0]
    adj_gm = GM(train_data)  # global correlation
    adj = get_adjacency_matrix(adj_filename,
                               num_of_vertices,
                               id_filename=id_filename)  # spatial correlation

    adj_dtw = np.array(pd.read_csv(adj_dtw_filename, header=None))

    adj_mx = construct_adj_fusion(adj, adj_gm, adj_dtw, 4, model_name)

    adj_st = torch.tensor(adj_mx, dtype=torch.float32)
    mask_init_value_st = (adj_mx != 0).astype('float32')
    log_string(log, 'adj shape:{}'.format(adj_st.shape))
    return adj_st.cuda(), mask_init_value_st


def GM(data):
    node = data.shape[1]
    A = np.zeros((node, node))
    error = np.zeros(A.shape)
    mean_arr = np.mean(data, axis=0)
    std_arr = np.std(data, axis=0)

    for i in range(0, node - 1):
        for j in range(i + 1, node):
            mean_std = (mean_arr[i] + mean_arr[j]) ** 2 + std_arr[i] ** 2 + std_arr[j] ** 2
            static = np.mean((data[:, i] + data[:, j]) ** 2)
            error[i][j] = error[j][i] = abs(mean_std - static)

        '''
        t3 = time.time()
        for i in range(0, node-1):
            for j in range(i + 1, node):
                compute_dtw(data[:, i], data[:, j])
        t4 = time.time()       
        print('the time of constructing dtw:', t4-t3)

        t5 = time.time()
        for i in range(0, node-1):
            for j in range(i + 1, node):
                PEARSONR(data[:, i], data[:, j])
        t6 = time.time()       
        print('the time of constructing pear:', t6-t5)
        '''

    Min = np.min(error)
    n = 4
    for i in range(error.shape[0]):
        for k in range(n):
            Max = np.max(error[i])
            index = np.argwhere(error[i] == Max)[0]
            A[i][index] = 1
            error[i][index] = Min

    for i in range(node):
        A[i][i] = 1

    return A


def PEARSONR(y_ob, y_pred):
    Cov = np.sum((y_pred - np.mean(y_pred)) * (y_ob - np.mean(y_ob)))
    Std1 = (np.sum((y_ob - np.mean(y_ob)) ** 2) ** 0.5)
    Std2 = (np.sum((y_pred - np.mean(y_pred)) ** 2) ** 0.5)
    return (Cov / (Std1 * Std2))


def compute_dtw(a, b, o=1, T=12):
    # a=normalize(a)
    # b=normalize(b)
    T0 = 288

    d = np.reshape(a, [-1, 1, T0]) - np.reshape(b, [-1, T0, 1])
    d = np.linalg.norm(d, axis=0, ord=o)
    D = np.zeros([T0, T0])
    for i in range(T0):
        for j in range(max(0, i - T), min(T0, i + T + 1)):
            if (i == 0) and (j == 0):
                D[i, j] = d[i, j] ** o
                continue
            if (i == 0):
                D[i, j] = d[i, j] ** o + D[i, j - 1]
                continue
            if (j == 0):
                D[i, j] = d[i, j] ** o + D[i - 1, j]
                continue
            if (j == i - T):
                D[i, j] = d[i, j] ** o + min(D[i - 1, j - 1], D[i - 1, j])
                continue
            if (j == i + T):
                D[i, j] = d[i, j] ** o + min(D[i - 1, j - 1], D[i, j - 1])
                continue
            D[i, j] = d[i, j] ** o + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    return D[-1, -1] ** (1.0 / o)


def get_adjacency_matrix(distance_df_filename, num_of_vertices,
                         type_='connectivity', id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    type_: str, {connectivity, distance}

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    import csv

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                A[id_dict[j], id_dict[i]] = 1

        for i in range(A.shape[0]):
            A[i][i] = 1

        return A

    # Fills cells in the matrix with distances.
    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == 'connectivity':
                A[i, j] = 1
                A[j, i] = 1
            elif type == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be "
                                 "connectivity or distance!")

    for i in range(A.shape[0]):
        A[i][i] = 1

    return A


def construct_adj_fusion(adj_s, adj_gm, adj_dtw, steps, type='CSTL'):
    '''
    construct a bigger adjacency matrix using the given matrix

    Parameters
    ----------
    A: np.ndarray, adjacency matrix, shape is (N, N)

    steps: how many times of the does the new adj mx bigger than A

    Returns
    ----------
    new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)

    ----------
    This is 4N_1 mode:

    [T, 1, 1, T
     1, S, 1, 1
     1, 1, S, 1
     T, 1, 1, T]

    '''
    N = len(adj_s)
    adj_T = np.identity(N)
    adj_0 = np.zeros((N, N))

    adj = np.c_[adj_T, adj_T, adj_T, adj_T]
    #adj = np.c_[adj_0, adj_0, adj_T, adj_s]
    #adj = np.c_[adj_dtw, adj_T, adj_T, adj_dtw]
    #adj = np.c_[adj_gm, adj_T, adj_T, adj_gm]
    

    return adj



