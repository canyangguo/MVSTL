import time
import torch
import numpy as np
from utils.evaluation import masked_mae_np, masked_mape_np, masked_mse_np
from utils.logs import log_string
import random


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # for CPU
    torch.cuda.manual_seed(seed)  # for current GPU
    torch.cuda.manual_seed_all(seed)  # for all GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def param_init(net, log):
    num_params = 0

    for param in net.parameters():
        log_string(log, str(param.shape), pnt=False)
        num_params += param.numel()
    return num_params


def start_testing(net, test_loader, best_epoch, num_for_predict, num_of_vertices, t, log):
    net.eval()
    with torch.no_grad():
        pres = np.zeros((0, num_for_predict, num_of_vertices))
        labels = np.zeros((0, num_for_predict, num_of_vertices))
        for idx, (data, label) in enumerate(test_loader):
            data = data.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            pre = net(data)

            pres = np.r_[pres, pre.to('cpu').detach().numpy()]
            labels = np.r_[labels, label.to('cpu').detach().numpy()]

        tmp_info = []
        for idx in range(num_for_predict):
            y, x = labels[:, idx: idx + 1, :], pres[:, idx: idx + 1, :]
            tmp_info.append((
                masked_mae_np(y, x, 0),
                masked_mape_np(y, x, 0),
                masked_mse_np(y, x, 0) ** 0.5
            ))
        tmp_info.append((
            masked_mae_np(labels[:, : 12, :], pres[:, : 12, :], 0),
            masked_mape_np(labels[:, : 12, :], pres[:, : 12, :], 0),
            masked_mse_np(labels[:, : 12, :], pres[:, : 12, :], 0) ** 0.5
        ))
        mae, mape, rmse = tmp_info[-1]
        log_string(log, 'test: best epoch: {}, mae: {:.3f}, mape: {:.3f}, rmse: {:.3f}, time: {:.3f}s\n'.format(
            best_epoch, mae, mape, rmse, time.time() - t))
    return tmp_info


def start_training(net, ep, optimizer, criterion, mae_criterion, train_loader, training_samples, lamda, drop_noise, t, log):
    mae = 0
    net.train()
    with torch.enable_grad():
        for idx, (data, label) in enumerate(train_loader):
            data = data.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            pre = net(data)
            if drop_noise == True:

                error = torch.abs(pre - label)
                pos = torch.where(error < torch.mean(error) + lamda * torch.std(error))
                loss_tra = criterion(pre[pos], label[pos])
            else:
                loss_tra = criterion(pre, label)
            # print(error.shape, pos[0].shape[0], error.shape[0]*error.shape[1]*error.shape[2]-pos[0].shape[0])

            mae += mae_criterion(pre, label).item() * (pre.shape[0] / training_samples)
            optimizer.zero_grad()
            loss_tra.backward()
            optimizer.step()

    log_string(log, 'training: epoch: {}, mae: {:.3f}, time: {:.3f}s'.format(
        ep + 1, mae, time.time() - t))
    return net, mae


def valdiation(net,  ep, best_epoch, mae_criterion, val_loader, val_samples, t, lowest_val_loss, params_filename, log):
    net.eval()
    mae = 0
    with torch.no_grad():

        for idx, (data, label) in enumerate(val_loader):
            data = data.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            pre = net(data)

            mae += mae_criterion(pre, label).item() * (pre.shape[0] / val_samples)

        log_string(log, 'validation: epoch: {}, loss: {:.3f}, time: {:.3f}s'.format(
            ep + 1, mae, time.time() - t))

        if mae < lowest_val_loss:
            log_string(log, 'update params...\n')
            best_epoch = ep + 1
            torch.save(net.state_dict(), params_filename)
            lowest_val_loss = mae
    return best_epoch, lowest_val_loss, mae



def training(net, train_loader, val_loader, test_loader, epochs, training_samples, val_samples,
             learning_rate, num_for_predict, num_of_vertices, params_filename, lamda, log, wd=1e-5, drop_noise=True):
    # if torch.cuda.device_count() > 1:
    # net = net.cuda()
    # net = torch.nn.DataParallel(net.cuda(), device_ids=[0,1])  # n GPUs

    criterion = torch.nn.HuberLoss(delta=1)
    mae_criterion = torch.nn.L1Loss(reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=wd)

    lowest_val_loss = 1e6
    info_train, info_val = [], []
    best_epoch = 0

    for ep in range(epochs):
        if ep == 150:
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate * 0.1, weight_decay=wd)

        t = time.time()
        net, train_mae = start_training(net, ep, optimizer, criterion, mae_criterion, train_loader, training_samples, lamda, drop_noise, t, log)
        best_epoch, lowest_val_loss, val_mae = valdiation(net,  ep, best_epoch, mae_criterion, val_loader, val_samples, t, lowest_val_loss, params_filename, log)

        info_train.append(train_mae)
        info_val.append(val_mae)

    net.load_state_dict(torch.load(params_filename))
    tmp_info = start_testing(net, test_loader, best_epoch, num_for_predict, num_of_vertices, t, log)

    return tmp_info, info_train, info_val

