import os
import json
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
from data.component.rand_augment import RandAugment
from data.component.utils import IMAGE_MEAN, IMAGE_STD
import mindformers.models.clip.clip_tokenizer


class COCODataset:
    def __init__(self, root_path, annotation_path, need_type='all', train=True):
        #from clip import tokenize
        self.need_type = need_type
        self.train = train
        self.tokenizer = tokenize
        self.img_mean, self.img_std = IMAGE_MEAN, IMAGE_STD

        self.trans = [
            CV.Resize(size=(224, 224)),
            CV.CenterCrop(size=(224, 224)),
            RandAugment(num_ops=4),
            CV.ToTensor(),
            CV.Normalize(mean=self.img_mean, std=self.img_std)
        ] if train else [
            CV.Resize(size=(224, 224)),
            CV.CenterCrop(size=(224, 224)),
            CV.ToTensor(),
            CV.Normalize(mean=self.img_mean, std=self.img_std)
        ]

        self.data_dir = os.path.join(root_path, 'train2017' if train else 'val2017')
        self.annotation_file = os.path.join(annotation_path,
                                            'captions_train2017.json' if train else 'captions_val2017.json')

        self.dataset = ds.CocoCaptionsDataset(self.data_dir, self.annotation_file)

    def __getitem__(self, idx):
        item = next(self.dataset.create_dict_iterator(output_numpy=True, num_epochs=1))
        image, caption = item['image'], item['caption'][0]
        image = image.transpose(2, 0, 1)  # MindSpore requires channels-first format
        caption = self.tokenizer(caption, truncate=False)[0]

        if self.need_type == 'all' or not self.train:
            return image, caption
        elif self.need_type == 'image':
            return image
        elif self.need_type == 'text':
            return caption
        else:
            raise ValueError('the mscoco dataset need_type parameter should be [\'all\', \'text\', \'image\'], '
                             f'but got {self.need_type}')