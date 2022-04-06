#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import argparse

def get_args():

    parser = argparse.ArgumentParser(description='PyTorch for DUL-regression')
    
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=0)

    # 训练参数
    parser.add_argument('--dataset', type = str, default = 'mnist', help = 'mnist, fashion_mnist')
    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)

    # 优化器参数
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--milestones', type=list, default=[10, 20])
    parser.add_argument('--scheduler_gama', type=float, default=0.2)

    args = parser.parse_args()

    return args
