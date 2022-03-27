#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import argparse

def get_args():

    parser = argparse.ArgumentParser(description='PyTorch for DUL-regression')
    
    parser.add_argument('--debug', type=bool, default=True)

    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument('--dataset', type = str, default = 'cifar10', help = 'cifar10')

    args = parser.parse_args()

    return args
