import os
from typing import List
from typing import Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split


import mindspore.dataset as ds
import mindspore.dataset.vision as CV
import mindspore.dataset.transforms as T

import mindspore.dataset.vision.transforms as C
from mindspore.dataset.vision.transforms import Normalize
from mindspore import Tensor

from .component.utils import IMAGE_MEAN, IMAGE_STD

class TextImageDataModule:
    def __init__(self, image_path, batch_size=64, workers=4):
        super(TextImageDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = workers
        print("batch_size", self.batch_size, "num_workers", self.num_workers)
        self.img_mean = IMAGE_MEAN
        self.img_std = IMAGE_STD
        url = [str(i) for i in list(Path(image_path).glob('*.tar'))]
        self.train_url, self.val_url = self.train_val_split(image_path)
        print(f'len(train) == {len(self.train_url)}, len(val) == {len(self.val_url)}')

    def train_val_split(self, image_path: str) -> Tuple[str]:
        files = os.listdir(image_path)
        train_files, val_files = train_test_split(files, test_size=0.1)
        train_urls = [os.path.join(image_path, file) for file in train_files]
        val_urls = [os.path.join(image_path, file) for file in val_files]
        return train_urls, val_urls

    def make_transform(self, is_train):
        self.img_mean=IMAGE_MEAN
        self.img_std=IMAGE_STD
        transform_list = [
            CV.Resize(224),
            CV.CenterCrop(224),
            #CV.Rescale(1.0 / 255.0, 0.0)
        ]
        if is_train:
            transform_list.append(CV.RandAugment(num_ops=4))
        transform_list.extend([
            #CV.ToTensor(),
            CV.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
            CV.HWC2CHW()
        ])
        return T.Compose(transform_list)

    def make_loader(self, is_train):
        if is_train:
            urls = self.train_url
            #dataset_size = 551335
            shuffle = 5000
        else:
            urls = self.val_url
            #dataset_size = 64376
            shuffle = 0

        transform = self.make_transform(is_train)

        dataset = ds.GeneratorDataset(urls, ["jpg", "txt"])

        dataset = dataset.shuffle(shuffle)
        dataset = dataset.map(operations=transform, input_columns=["jpg"], num_parallel_workers=self.num_workers)

        if is_train:
            dataset = dataset.batch(self.batch_size, drop_remainder=True)
        else:
            dataset = dataset.batch(self.batch_size)

        return dataset

    def train_dataloader(self):
        return self.make_loader(is_train=True)

    def val_dataloader(self):
        return self.make_loader(is_train=False)