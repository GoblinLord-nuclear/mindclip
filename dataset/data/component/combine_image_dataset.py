import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image
import mindspore.dataset as ds
import mindspore.dataset.transforms.transforms as C
import mindspore.dataset.vision.transforms as CV
from mindspore import Tensor

from .rand_augment import RandAugment
from .utils import IMAGE_DATASET_NAME, IMAGE_PREFIX, IMAGE_MEAN, IMAGE_STD, encode_texts

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)


def prepare(prepare_args):
    raw_data_dir = Path(prepare_args['raw_data_dir'])
    cache_dir = Path(prepare_args['cache_dir'])
    teacher_name = prepare_args['teacher_name']
    overwrite = prepare_args['overwrite']

    cache_path = cache_dir / f'image-cache-val-{teacher_name.replace("/", "-")}.npz'
    if not cache_path.exists() or overwrite:
        logging.info('the cache_dir not exists or you set overwrite')
        val_image_file_list_path = raw_data_dir / 'mscoco' / 'val2017'
        path_list = []
        captions = []
        annotations_dir = raw_data_dir / 'mscoco' / 'annotations'
        with open((annotations_dir / 'captions_val2017.json'), 'r') as f:
            coco_data = json.load(f)
        images = coco_data['images']
        id2caption = {}
        id2filename = {}
        for image in images:
            id2filename[image['id']] = image['file_name']
        for annotation in coco_data['annotations']:
            id2caption[annotation['image_id']] = annotation['caption']
        for id, file_name in id2filename.items():
            caption = id2caption.get(id, None)
            if caption:
                captions.append(caption)
                path_list.append(val_image_file_list_path / file_name)

        captions_rep = encode_texts(captions, teacher_name)
        np.savez(cache_path, path_list=path_list, captions_rep=captions_rep, captions=captions)
        logging.info(f'cache data saved in {str(cache_path)}')


class CombineImageDataset:
    def __init__(self, combine_dataset_path, train=True, image_use=None, cache_dir='cache', teacher_name='ViT-B/32'):
        if image_use is None:
            image_use = ['coco', 'data_256', 'imagenet']

        for i in image_use:
            assert i in IMAGE_DATASET_NAME, f'the {i} dataset name is not exists in {IMAGE_DATASET_NAME}'
        self.train = train
        self.img_mean = IMAGE_MEAN
        self.img_std = IMAGE_STD
        self.teacher_name = teacher_name

        cache_path = Path(cache_dir) / f'image-cache-val-{teacher_name.replace("/", "-")}.npz'
        if not train:
            data = np.load(cache_path)
            self.path_list, self.captions_rep, self.captions = data['path_list'], data['captions_rep'], data['captions']
        else:
            self.train_image_file_path = Path(combine_dataset_path)

            def filter_dataset(x):
                res = False
                for name in image_use:
                    prefix = IMAGE_PREFIX[name]
                    res = res or x.startswith(prefix)
                return res

            self.path_list = [path for path in self.train_image_file_path.iterdir() if filter_dataset(path.name)]

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        path = self.path_list[idx]
        img = Image.open(path).convert('RGB')

        trans = [
            RandAugment(num_ops=4),
            CV.ToTensor(),
            CV.Normalize(self.img_mean, self.img_std),
        ] if self.train else [
            CV.Resize((224, 224)),
            CV.ToTensor(),
            CV.Normalize(self.img_mean, self.img_std)
        ]

        transform = C.Compose(trans)

        img = transform(img)

        if self.train:
            return img
        else:
            return img, self.captions_rep[idx], self.captions[idx]