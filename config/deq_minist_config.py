#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import argparse

def get_args():

    parser = argparse.ArgumentParser(description='PyTorch for DUL-regression')
    
    # -- env
    
    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument('--dataset', type = str, default = 'fashion_mnist', help = 'mnist, fashion_mnist, cifar10, or cifar100')

    args = parser.parse_args()

    return args
