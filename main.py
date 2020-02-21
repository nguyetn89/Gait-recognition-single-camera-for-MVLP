"""
Gait recognition for MVLP gait dataset
Licence: BSD 2-Clause "Simplified"
Author : Trong Nguyen Nguyen
"""

import os
import argparse
from model import NetworkController

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', help='task', default=None)
    parser.add_argument('-e', '--epoch', help='epoch range', default=None)
    parser.add_argument('-s', '--scale', help='scaling input size', default=1)
    parser.add_argument('-a', '--angle', help='angle', default=None)
    parser.add_argument('-b', '--batch', help='batch size', default=8)
    parser.add_argument('-r', '--random_pair', help='randomizing training data of diff-pairs', default=1)
    args = vars(parser.parse_args())
    # variable
    task = args['task']
    assert task in ("training", "evaluation")
    epoch = args['epoch']
    if '-' in epoch:
        vals = epoch.split('-')
        assert len(vals) == 2
        epoch = [int(val) for val in vals]
    else:
        epoch = int(epoch)
    scale = float(args["scale"])
    angle = int(args["angle"])
    batch_size = int(args["batch"])
    random_pair = bool(int(args["random_pair"]))
    # config
    data_path = "./gait_datasets/GaitMVLP/GEI"
    store_path = "./storage_random_impair"
    input_size = (int(128*scale), int(88*scale))
    # init network
    print("Network init...")
    controller = NetworkController(input_size, data_path, angle, store_path)
    # train model
    if task == "training":
        if isinstance(epoch, int):
            print("Network training epochs %d-%d..." % (0, epoch))
            controller.train(0, epoch, batch_size=batch_size, save_every_x_epochs=10,
                             random_diff_GEI_pairs=random_pair)
        else:
            print("Network training epochs %d-%d..." % (epoch[0], epoch[1]))
            controller.train(epoch[0], epoch[1], batch_size=batch_size, save_every_x_epochs=10,
                             random_diff_GEI_pairs=random_pair)
    # test model
    elif task == "evaluation":
        assert isinstance(epoch, int)
        print("Network testing epoch %d..." % epoch)
        controller.eval(epoch, batch_size=batch_size, angle=angle)
