#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import argparse

def get_args():

    parser = argparse.ArgumentParser(description='PyTorch for DUL-regression')
    
    # -- env
    
    parser.add_argument('--n_epoch', type=int, default=200)

    
    args = parser.parse_args()

    return args
