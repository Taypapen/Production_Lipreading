import os
import numpy as np
import datetime
import logging
import math
import torch.optim as optim


class AverageTracker(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def change_lr_on_optimizer(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class CosineScheduler:
    def __init__(self, lr_ori, epochs):
        self.lr_ori = lr_ori
        self.epochs = epochs

    def adjust_lr(self, optimizer, epoch):
        reduction_ratio = 0.5 * (1 + math.cos(math.pi * epoch / self.epochs))
        change_lr_on_optimizer(optimizer, self.lr_ori*reduction_ratio)


def showLR(optimizer):
    return optimizer.param_groups[0]['lr']


def get_optimizer(type, optim_policies, lr):
    # -- define optimizer
    if type == 'adam':
        optimizer = optim.Adam(optim_policies, lr=lr, weight_decay=1e-4)
    elif type == 'adamw':
        optimizer = optim.AdamW(optim_policies, lr=lr, weight_decay=1e-2)
    elif type == 'sgd':
        optimizer = optim.SGD(optim_policies, lr=lr, weight_decay=1e-4, momentum=0.9)
    else:
        raise NotImplementedError
    return optimizer


def get_wordslist_from_txt_file(file_path):
    with open(file_path) as file:
        word_list = file.readlines()
        word_list = [item.rstrip() for item in word_list]
    return word_list

def read_txt_lines(filepath):
    assert os.path.isfile( filepath ), "Error when trying to read txt file, path does not exist: {}".format(filepath)
    with open( filepath ) as myfile:
        content = myfile.read().splitlines()
    return content

def logging_setup(args, save_path):
    if args is None:
        log_path = '{}/logging.txt'.format(save_path)
    else:
        log_path = '{}/{}_{}_{}classes_log.txt'.format(save_path, args.model_version, args.lr, args.num_classes)
    logger = logging.getLogger("the_logs")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    return logger

def get_save_folder(args):
    # create save and log folder
    save_path = '{}/{}'.format(args.logging_direc, args.model_version)
    save_path += '/' + datetime.datetime.now().isoformat().split('.')[0]
    save_path = save_path.replace(':','.')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    return save_path
