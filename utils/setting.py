import numpy as np
import torch
import random
import argparse

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # for CPU
    torch.cuda.manual_seed(seed)  # for current GPU
    torch.cuda.manual_seed_all(seed)  # for all GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False




def log_string(log, string, p=True):  # p decide print
    log.write(string + '\n')
    log.flush()
    if p:
        print(string)


def parser_set():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help='configuration file')
    parser.add_argument("--test", default=False, action="store_true", help="test program")
    parser.add_argument("--plot", help="plot network graph", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--gpu", type=str, default='0', help="gpu ID")
    parser.add_argument("--st_dropout_rate", type=float, default=0.95, help="st_dropout_rate")
    parser.add_argument("--st_dropout_noise_rate", type=float, default=4, help="st_dropout_rate")
    parser.add_argument("--model_name", type=str, default='new', help="model_name")
    parser.add_argument("--num_of_latents", type=int, default=32, help="num_of_latents")
    args = parser.parse_args()
    return args


def select_GPU(ctx):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = parser_set().gpu
    #return torch.device(ctx if torch.cuda.is_available() else 'cpu')

