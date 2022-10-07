# -*- coding:utf-8 -*-
import json
from utils.data_process import load_data, load_adj
from utils.model_fit import setup_seed, training, param_init
from utils.logs import log_string
import numpy as np
import torch
import random
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file')
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--gpu", type=str, default='0', help="gpu ID")
parser.add_argument("--st_dropout_rate", type=float, default=0.95, help="st_dropout_rate")
parser.add_argument("--noise_rate", type=float, default=4, help="st_dropout_rate")
parser.add_argument("--missing_rate", type=float, default=0.)
parser.add_argument("--model_name", type=str, default='TS', help="model_name")
parser.add_argument("--num_of_latents", type=int, default=32, help="num_of_latents")
parser.add_argument("--test", default=False)
args = parser.parse_args()

config_filename = args.config
with open(config_filename, 'r') as f:
    config = json.loads(f.read())

model_name = args.model_name
graph_signal_matrix_filename = config['graph_signal_matrix_filename']
adj_dtw_filename = config['adj_dtw_filename']
adj_filename = config['adj_filename']
num_of_vertices = config['num_of_vertices']
id_filename = config['id_filename']
batch_size = config['batch_size']
points_per_hour = config['points_per_hour']
d_model = config['d_model']
filters = config['filters']
use_mask = config['use_mask']
temporal_emb = config['temporal_emb']
spatial_emb = config['spatial_emb']
num_for_predict = config['num_for_predict']
num_of_features = config['num_of_features']
receptive_length = config['receptive_length']
num_of_gcn_filters = config['num_of_gcn_filters']
st_dropout_rate = args.st_dropout_rate
num_of_latents = args.num_of_latents
epochs = config['epochs']
learning_rate = config['learning_rate']
missing_rate = args.missing_rate
noise_rate = args.noise_rate
config_name = '_' + args.model_name + '_' + str(noise_rate) + '_' + str(st_dropout_rate) + '_' + str(num_of_latents)

log = open(config['log_path'] + config_name + '.txt', 'w')
param_file = config['params_filename'] + config_name

if args.model_name == 'ours':
    from models.CSTL import make_model

log_string(log, 'let us begin! traning ' + model_name + ' ○（*￣︶￣*）○\n')
log_string(log, 'param file: {}'.format(param_file))

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
setup_seed(args.seed)
log_string(log, 'gpu ID: {}, random seed: {}\n'.format(args.gpu, args.seed))

log_string(log, str(json.dumps(config, sort_keys=True, indent=4)), pnt=False)

log_string(log, '*****data loading*****\n')
adj_st, mask_init_value_st = load_adj(adj_dtw_filename, graph_signal_matrix_filename, adj_filename, num_of_vertices, id_filename, model_name, log)
train_loader, val_loader, test_loader, training_samples, val_samples, test_sample = load_data(graph_signal_matrix_filename, batch_size, args.test, log, missing_rate)

log_string(log, '*****make model*****\n')
net = make_model(adj_st, points_per_hour, num_of_vertices, d_model, filters, use_mask, mask_init_value_st,
            temporal_emb, spatial_emb, num_for_predict, num_of_features, receptive_length, st_dropout_rate,
            num_of_latents, num_of_gcn_filters).cuda()
num_params = param_init(net, log)
log_string(log, "num of parameters: {}".format(num_params))

log_string(log, '*****start training model*****')
all_info, train_loss, val_loss = training(net, train_loader, val_loader, test_loader, epochs,
                                          training_samples, val_samples, learning_rate, num_for_predict,
                                          num_of_vertices, param_file, noise_rate, log)

log_string(log, '*****loss curve*****')
log_string(log, 'train_loss:\n' + str(train_loss))
log_string(log, 'val_loss:\n' + str(val_loss))

log_string(log, '*****multi step prediction*****')
for i in all_info:
    log_string(log, '{:.2f} {:.2f} {:.2f}'.format(*i))

log.close()



