"""
CNN model for gait recognition
Licence: BSD 2-Clause "Simplified"
Author : Trong Nguyen Nguyen
"""

import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils import MVLPdataset, ProgressBar
torch.manual_seed(3011)

LEN_ZFILL = 4


# https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


# https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
class InceptionA(nn.Module):

    def __init__(self, in_channels, list_channels, conv_block=None):
        super(InceptionA, self).__init__()
        self.in_channels = in_channels
        self.list_channels = list_channels

        channels_1x1, channels_5x5, channels_3x3, channels_pooling = list_channels
        assert isinstance(channels_1x1, int)
        assert isinstance(channels_5x5, (list, tuple)) and len(channels_5x5) == 2
        assert isinstance(channels_3x3, (list, tuple)) and len(channels_3x3) == 3
        assert isinstance(channels_pooling, int)
        self.n_out_channels = channels_1x1 + channels_5x5[-1] + channels_3x3[-1] + channels_pooling

        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, channels_1x1, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, channels_5x5[0], kernel_size=1)
        self.branch5x5_2 = conv_block(channels_5x5[0], channels_5x5[1], kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, channels_3x3[0], kernel_size=1)
        self.branch3x3dbl_2 = conv_block(channels_3x3[0], channels_3x3[1], kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(channels_3x3[1], channels_3x3[2], kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, channels_pooling, kernel_size=1)

    def get_n_out_channels(self):
        return self.n_out_channels

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)   # 64 channels

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)     # 64 channels

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)    # 96 channels

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)     # pool_features channels

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


def calc_conv_output_shape(input_size, kernel_size, stride, padding):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    if isinstance(padding, int):
        padding = [padding, padding]
    if isinstance(stride, int):
        stride = [stride, stride]
    return (np.array(input_size) - np.array(kernel_size) + 2*np.array(padding))//np.array(stride) + 1


class GaitNetwork(nn.Module):
    def __init__(self, input_size=(128, 88)):
        super().__init__()
        # set device & input shape
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size

        # first InceptionA: (h, w) -> (h, w)
        channels_Inception_b1 = [32, (32, 64), (32, 64, 64), 32]
        self.Inception_1 = InceptionA(2, channels_Inception_b1)
        out_channels_b1 = self.Inception_1.get_n_out_channels()
        # conv for pooling: (h, w) -> (h/2, w/2)
        self.conv_pooling_1 = nn.Sequential(nn.Conv2d(out_channels_b1,
                                                      out_channels_b1,
                                                      kernel_size=3,
                                                      stride=2,
                                                      padding=1),
                                            nn.ReLU())
        out_size_1 = calc_conv_output_shape(input_size, 3, 2, 1)

        # second InceptionA: (h/2, w/2) -> (h/2, w/2)
        channels_Inception_b2 = [64, (64, 128), (64, 128, 128), 64]
        self.Inception_2 = InceptionA(out_channels_b1, channels_Inception_b2)
        out_channels_b2 = self.Inception_2.get_n_out_channels()
        # conv for pooling: (h/2, w/2) -> (h/4, w/4)
        self.conv_pooling_2 = nn.Sequential(nn.Conv2d(out_channels_b2,
                                                      out_channels_b2,
                                                      kernel_size=3,
                                                      stride=2,
                                                      padding=1),
                                            nn.ReLU())
        out_size_2 = calc_conv_output_shape(out_size_1, 3, 2, 1)

        # last sequence: (h/4, w/4) -> ...
        n_flattened_units = out_channels_b2 * out_size_2[0] * out_size_2[1]
        self.logit = nn.Sequential(nn.Flatten(),
                                   nn.Linear(n_flattened_units, 2048),
                                   nn.ReLU(),
                                   nn.Linear(2048, 1))
        self.output = nn.Sigmoid()
        self.to(self.device)

    def forward(self, x):
        Incep_features_1 = self.Inception_1(x)
        features_1 = self.conv_pooling_1(Incep_features_1)
        Incep_features_2 = self.Inception_2(features_1)
        features_2 = self.conv_pooling_2(Incep_features_2)
        logit = self.logit(features_2)
        prob = self.output(logit)
        return logit, prob, [Incep_features_1, features_1, Incep_features_2, features_2]


class NetworkController(object):
    def __init__(self, input_size, data_path, angle, store_path, device_str=None):
        # paths
        self._input_size = input_size
        self._angle = angle
        self._data_path = data_path
        self._store_path = store_path + "/height_%s_width_%s_angle_%s" % \
            (str(input_size[0]).zfill(3), str(input_size[1]).zfill(3), str(angle).zfill(3))
        self._model_store_path = self._store_path + "/models"             # trained models
        self._output_store_path = self._store_path + "/outputs"           # inference outputs
        self._log_path = self._store_path + "/log"                        # tensorboard log
        self._create_all_paths()
        # device
        if device_str is None:
            self._device = torch.device("cuda:0" if torch.cuda.is_available()
                                        else "cpu")
        else:
            assert isinstance(device_str, str)
            self._device = torch.device(device_str)

        self._network = GaitNetwork(self._input_size)

        # ADAM optimizer
        self._learning_rate = 1e-4
        self._criterion = nn.BCEWithLogitsLoss()
        self._optimizer = optim.Adam(self._network.parameters(), lr=self._learning_rate)

        # Set the logger
        self._logger = SummaryWriter(self._log_path)
        self._logger.flush()

    # create necessary directories
    def _create_all_paths(self):
        def _create_path(path):
            if not os.path.exists(path):
                os.makedirs(path)
        _create_path(self._model_store_path)
        _create_path(self._output_store_path)
        _create_path(self._log_path)

    # load pretrained model and optimizer
    def _load_model(self, model_filename):
        self._network.load_state_dict(torch.load(os.path.join(self._model_store_path, model_filename)))
        print("Network loaded from %s" % model_filename)

    # save pretrained model and optimizer
    def _save_model(self, model_filename, print_msg=True):
        torch.save(self._network.state_dict(), os.path.join(self._model_store_path, model_filename))
        if print_msg:
            print("Network saved to %s" % model_filename)

    # statistic of numbers of samples for each class
    def _stat_samples(self, list_classes):
        classes = torch.unique(list_classes)
        print(len(classes), len(list_classes))

    # dataset: instance of GaitDataset
    def train(self, epoch_start, epoch_end, batch_size=8, save_every_x_epochs=2, random_diff_GEI_pairs=True):
        # set mode for network
        self._network.train()
        if epoch_start > 0:
            self._load_model("model_epoch_%s.pkl" % str(epoch_start).zfill(LEN_ZFILL))

        # turn on debugging related to gradient
        torch.autograd.set_detect_anomaly(True)

        # create data loader for yielding batches
        dataset = MVLPdataset(self._data_path, self._angle, self._input_size)
        dataset.load_data("train")
        dataset.set_mode("train")

        # create data loader
        dataset.prepare_training_data()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
        n_batch = int(np.ceil(len(dataset) / batch_size))

        print("Started time:", datetime.datetime.now())
        progress = ProgressBar(n_batch * (epoch_end - epoch_start), fmt=ProgressBar.FULL)

        # training for each epoch
        for epoch in range(epoch_start, epoch_end):
            iter_count = 0

            # process a batch
            for data in dataloader:
                self._optimizer.zero_grad()
                samples, labels = data
                samples, labels = samples.to(self._device), labels.to(self._device)
                # print(labels)
                out_logit, _, _ = self._network(samples)
                loss = self._criterion(out_logit, torch.unsqueeze(labels, 1))
                loss.backward()
                self._optimizer.step()

                msg = " [epoch %3d/%d - iter %4d/%d -> loss = %.4f]" \
                      % (epoch + 1, epoch_end, iter_count + 1, n_batch, loss.item())

                progress.current += 1
                progress(msg)

                info = {
                   'Loss': loss.item()
                }
                idx = epoch * n_batch + iter_count
                for tag, value in info.items():
                    self._logger.add_scalar(tag, value, idx)

                iter_count += 1

            # Saving model and sampling images every X epochs
            if (epoch + 1) % save_every_x_epochs == 0:
                self._save_model("model_epoch_%s.pkl" % str(epoch + 1).zfill(LEN_ZFILL), print_msg=False)

            # Generate indices for diff-pairs of GEIs in training data
            if random_diff_GEI_pairs:
                dataset.prepare_training_data(force_to_calc=True)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

        # finish iteration
        progress.done()
        print("Finished time:", datetime.datetime.now())

        # Save the trained parameters
        if (epoch + 1) % save_every_x_epochs != 0:  # not already saved inside loop
            self._save_model("model_epoch_%s.pkl" % str(epoch + 1).zfill(LEN_ZFILL))

    # dataset: instance of GaitDataset
    def eval(self, epoch, batch_size=8, angle=None):
        # load pretrained model and set to eval() mode
        self._load_model("model_epoch_%s.pkl" % str(epoch).zfill(LEN_ZFILL))
        self._network.eval()
        #
        if angle is None:
            angle = self._angle
        print("================== Evaluation for angle %d ==================" % angle)
        out_file = self._output_store_path + '/epoch_%s.pt' % str(epoch).zfill(LEN_ZFILL)
        if os.path.exists(out_file):
            results = torch.load(out_file)
        else:
            results = {"probs": {}, "true_positive": 0}

        # dataloader for yielding batches
        dataset = MVLPdataset(self._data_path, angle, self._input_size)
        dataset.load_data("test")
        dataloader = None
        test_IDs = dataset.get_available_test_IDs()

        print("Started time:", datetime.datetime.now())
        progress = ProgressBar(len(test_IDs), fmt=ProgressBar.FULL)
        for test_id in test_IDs:
            if test_id in results["probs"]:
                continue
            dataset.set_mode("test:" + str(test_id))
            dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)

            # init variable for evaluation
            tmp_results, tmp_labels = [], []

            with torch.no_grad():
                for data in dataloader:
                    samples, labels = data
                    samples, labels = samples.to(self._device), labels.to(self._device)
                    _, out_prob, _ = self._network(samples)
                    tmp_results.append(out_prob)
                    tmp_labels.append(labels)

            results["probs"][test_id] = torch.cat(tmp_results, dim=0)
            tmp_labels = torch.cat(tmp_labels, dim=0)
            _, idx_max_score = results["probs"][test_id].max(0)
            results["true_positive"] += int(tmp_labels[idx_max_score].data.item() == 1)

            # save current results to file
            torch.save(results, out_file)
            progress.current += 1
            progress()

        progress.done()

        print("Finished time:", datetime.datetime.now())
        print("Rank 1 accuracy = %.3f" % (results["true_positive"]/len(test_IDs)))
