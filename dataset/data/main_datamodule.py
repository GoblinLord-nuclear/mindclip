import importlib
import inspect
import os

import mindspore.dataset as ds
import mindspore.dataset.vision.py_transforms as py_transforms
import mindspore.dataset.transforms.transforms as C
#import mindspore.dataset.transforms.py_transforms as P
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.initializer import Normal
from mindspore.dataset.transforms import vision
from mindspore.dataset.vision import Inter
from mindspore.ops import operations as ops
from mindspore.train.callback import Callback
from mindspore.train.model import Model

class MainDataModule:
    def __init__(self,
                 dataset_para,
                 dataset_name,
                 dataset,
                 prepare_para=None,
                 num_workers=8,
                 train_batch_size=128,
                 val_batch_size=1250):

        super().__init__()
        self.num_workers = num_workers
        self.dataset = dataset
        self.dataset_para = dataset_para
        self.dataset_name = dataset_name
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.data_module = self.load_data_module()
        self.prepare_function = self.load_prepare()
        self.prepare_function_args = prepare_para
        if self.prepare_function_args:
            self.prepare_function_args.update(dataset_para)

    def prepare_data(self):
        if self.prepare_function:
            self.prepare_function(self.prepare_function_args)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.trainset = self.instancialize(train=True)
            self.valset = self.instancialize(train=False)
        if stage == 'test' or stage is None:
            self.testset = self.instancialize(train=False)

    def train_dataloader(self):
        return ds.GeneratorDataset(self.trainset, ['data', 'label'], shuffle=True)

    def val_dataloader(self):
        return ds.GeneratorDataset(self.valset, ['data', 'label'], shuffle=False)

    def test_dataloader(self):
        return ds.GeneratorDataset(self.testset, ['data', 'label'], shuffle=False)

    def load_prepare(self):
        dataset_file = self.dataset
        module = importlib.import_module("component." + dataset_file)
        if hasattr(module, 'prepare'):
            prepare_function = getattr(module, 'prepare')
        else:
            prepare_function = None
        return prepare_function

    def load_data_module(self):
        dataset_file = self.dataset
        name = self.dataset_name
        try:
            data_module = getattr(importlib.import_module("component." + dataset_file), name)
        except:
            raise ValueError(f'Invalid Dataset File Name or Invalid Class Name data.{dataset_file}.{name}')
        return data_module

    def instancialize(self, **other_args):
        class_args = inspect.signature(self.data_module.__init__).parameters
        inkeys = self.dataset_para.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.dataset_para[arg]
        args1.update(other_args)
        return self.data_module(**args1)