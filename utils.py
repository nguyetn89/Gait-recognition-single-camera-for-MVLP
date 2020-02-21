"""
Utilities for gait recognition
Licence: BSD 2-Clause "Simplified"
Author : Trong Nguyen Nguyen
"""

import torch
import os
import sys
import re
import numpy as np
import torchvision.transforms as transforms

from PIL import Image

np.random.seed(3011)

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

tensor_normalize = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean=mean,
                                                            std=std)])


# Class forming dataset helper
# data access: self._data["train"/"test"]["probe"/"gallery"]["data"/"id"] (tensor)
class MVLPdataset(torch.utils.data.Dataset):
    def __init__(self, GEI_path, angle, img_size):
        super().__init__()
        # set attributes
        self._GEI_path = GEI_path
        self._img_size = img_size
        self._Resizer = transforms.Resize(self._img_size, interpolation=2)
        self._angle = angle

        # for other functions accessing
        self._IDs = {"train": list(range(1, 10306, 2)),
                     "test": list(range(2, 10307, 2)) + [10307]}
        self._common_ids_train = None
        self._indices_same_class = None

        self.data = {}      # data for getting batch
        self.indices = {}   # indices for getting batch
        self.labels = {}   # labels for getting batch

    def get_available_test_IDs(self):
        return self.data["test"]["probe"]["id"]

    def _load_data_subset(self, path, IDs):
        loaded_data, loaded_ids = [], []
        for id in IDs:
            if os.path.exists(path + "/%s.png" % str(id).zfill(5)):
                loaded_data.append(tensor_normalize(self._Resizer(Image.open(path + "/%s.png" % str(id).zfill(5)))))
                loaded_ids.append(id)
        print("loaded %d/%d GEIs" % (len(loaded_data), len(IDs)))
        return {"data": torch.cat(loaded_data, dim=0), "id": loaded_ids}

    def load_data(self, part):
        assert isinstance(part, str) and part in ("train", "test")
        out_file = os.path.join(self._GEI_path, "%s_%s_%d_x_%d.pt" %
                                (str(self._angle).zfill(3), part, self._img_size[0], self._img_size[1]))
        if os.path.exists(out_file):
            self.data[part] = torch.load(out_file)
        else:
            self.data[part] = {}
            self.data[part]["probe"] = self._load_data_subset(self._GEI_path + "/%s-00" % str(self._angle).zfill(3), self._IDs[part])
            self.data[part]["gallery"] = self._load_data_subset(self._GEI_path + "/%s-01" % str(self._angle).zfill(3), self._IDs[part])
            torch.save(self.data[part], out_file)

    # mode: "train" or "test:an_ID_probe"
    def set_mode(self, mode):
        assert isinstance(mode, str)
        assert mode == "train" or (mode[:5] == "test:" and int(mode[5:]) in self.data["test"]["probe"]["id"])
        self._mode = mode

    # indices["train"]["same_class"/"diff_class"]: list of (idx_probe, idx_gallery)
    def prepare_training_data(self, force_to_calc=False):
        if "train" not in self.indices or force_to_calc:
            if "train" not in self.data:
                self.load_data("train")
            # indices of same class GEI pairs (length = n)
            if self._common_ids_train is None or self._indices_same_class is None:
                self._common_ids_train = set(self.data["train"]["probe"]["id"]).intersection(self.data["train"]["gallery"]["id"])
                idx_common_probe = [self.data["train"]["probe"]["id"].index(i) for i in self._common_ids_train]
                idx_common_gallery = [self.data["train"]["gallery"]["id"].index(i) for i in self._common_ids_train]
                self._indices_same_class = list(zip(idx_common_probe, idx_common_gallery))

            # indices of the others (up to n)
            indices_diff_class = \
                list(zip(np.random.permutation(np.arange(len(self.data["train"]["probe"]["id"])))[:len(self._common_ids_train)],
                         np.random.permutation(np.arange(len(self.data["train"]["gallery"]["id"])))[:len(self._common_ids_train)]))

            # make sure two indices subsets are disjoint
            common_indices = set(self._indices_same_class).intersection(indices_diff_class)
            while len(common_indices) > 0:
                for common_idx in common_indices:
                    indices_diff_class.remove(common_idx)
                    indices_diff_class.append((np.random.randint(len(self.data["train"]["probe"]["id"])),
                                               np.random.randint(len(self.data["train"]["gallery"]["id"]))))
                common_indices = set(self._indices_same_class).intersection(indices_diff_class)

            # concat indices and generate labels
            labels_same_class = np.ones(len(self._indices_same_class), dtype=float)
            labels_diff_class = np.zeros(len(indices_diff_class), dtype=float)
            self.indices["train"] = np.array(self._indices_same_class + indices_diff_class, dtype=int)
            self.labels["train"] = np.array(list(labels_same_class) + list(labels_diff_class), dtype=float)
            permu = np.random.permutation(np.arange(len(self.labels["train"])))
            self.indices["train"] = self.indices["train"][permu]
            self.labels["train"] = self.labels["train"][permu]
            assert len(np.unique(self.labels["train"])) == 2

        else:
            print("Information of training part was already loaded -> skip this step")

    def __getitem__(self, index):
        if self._mode == "train":
            # get index
            sample_idx = self.indices["train"][index]
            assert len(sample_idx) == 2
            # get corresponding data
            sample = torch.cat([torch.unsqueeze(self.data["train"]["probe"]["data"][sample_idx[0]], 0),
                                torch.unsqueeze(self.data["train"]["gallery"]["data"][sample_idx[1]], 0)], dim=0)
            label = self.labels["train"][index]
            # assert isinstance(label, int)
        else:
            test_id = int(self._mode[5:])
            sample_idx = self.data["test"]["probe"]["id"].index(test_id)
            sample = torch.cat([torch.unsqueeze(self.data["test"]["probe"]["data"][sample_idx], 0),
                                torch.unsqueeze(self.data["test"]["gallery"]["data"][index], 0)], dim=0)
            label = int(test_id == self.data["test"]["gallery"]["id"][index])
        return sample, label

    def __len__(self):
        if self._mode == "train":
            return len(self.indices["train"])
        else:
            return len(self.data["test"]["gallery"]["id"])


# Modified from https://stackoverflow.com/questions/3160699/python-progress-bar
class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

    def __init__(self, total, width=80, fmt=DEFAULT, symbol='#', output=sys.stderr):
        assert len(symbol) == 1
        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
                          r'\g<name>%dd' % len(str(total)), fmt)
        self.current = 0

    def __call__(self, msg=''):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '|' + self.symbol * size + '.' * (self.width - size) + '|'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        print('\r' + self.fmt % args + msg, file=self.output, end='')

    def done(self):
        self.current = self.total
        self()
        print('', file=self.output)
