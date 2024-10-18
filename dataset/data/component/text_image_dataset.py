import json
import os
from pathlib import Path
from PIL import Image
#from clip import tokenize
import mindformers.models.clip.clip_tokenizer
import numpy as np
from mindspore.dataset import GeneratorDataset
from mindspore import dtype as mstype
import mindspore.dataset.vision as CV

class TestImageDataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_list = os.listdir(file_path)
        self.img_mean = IMAGE_MEAN
        self.img_std = IMAGE_STD
        self.preprocess = CV.ComposeOp([
            CV.Resize((224, 224)),
            CV.CenterCrop((224, 224)),
            CV.Rescale(1.0 / 255.0, 0.0),
            CV.Normalize(self.img_mean, self.img_std)
        ])

    def __getitem__(self, item):
        photo_path = os.path.join(self.file_path, self.file_list[item])
        image = Image.open(photo_path).convert('RGB')
        #image = np.array(image)
        image = self.preprocess(image)
        return self.file_list[item], image

    def __len__(self):
        return len(self.file_list)

class TestDataset:
    def __init__(self, data_dir, preprocess):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenize
        self.img_mean = IMAGE_MEAN
        self.img_std = IMAGE_STD
        self.sentences, self.captions, self.path_list = self.process()
        self.preprocess = preprocess

    def process(self):
        val_image_file_list_path = os.path.join(self.data_dir, 'mscoco', 'val2017')
        path_list = []
        captions = []
        sentences = []
        file_dir = os.path.join(self.data_dir, 'mscoco', 'annotations', 'captions_val2017.json')
        with open(file_dir, 'r', encoding='utf8') as f:
            data = json.load(f)
        images = data['images']
        id2caption = {}
        id2filename = {}
        for image in images:
            id2filename[image['id']] = image['file_name']
        for annotation in data['annotations']:
            id2caption[annotation['image_id']] = annotation['caption']
        for id, file_name in id2filename.items():
            caption = id2caption.get(id, None)
            if caption:
                sentences.append(caption)
                captions.append(self.tokenizer(caption).squeeze())
                path_list.append(os.path.join(val_image_file_list_path, file_name))

        return sentences, captions, path_list

    def __getitem__(self, idx):
        path = self.path_list[idx]
        img = Image.open(path).convert('RGB')
        #img = np.array(img)
        img = self.preprocess(img)
        return os.path.basename(self.path_list[idx]), img, self.captions[idx], self.sentences[idx]

    def __len__(self):
        return len(self.path_list)

# 创建数据集对象
#test_image_dataset = TestImageDataset(data_dir)
#test_dataset = TestDataset(data_dir, preprocess)
# 创建GeneratorDataset对象,测试用
#ds_image = GeneratorDataset(test_image_dataset, ["filename", "image"], shuffle=False)
#ds = GeneratorDataset(test_dataset, ["filename", "image", "caption", "sentence"], shuffle=False)