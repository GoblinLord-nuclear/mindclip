import json
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
import mindspore.dataset as ds
import mindformers.models.clip.clip_tokenizer
from mindspore import Tensor

from .utils import encode_images, IMAGE_STD, IMAGE_MEAN

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)


def prepare(prepare_args):
    #from clip import tokenize
    cache_dir = Path(prepare_args['cache_dir'])
    raw_data_dir = Path(prepare_args['raw_data_dir'])
    teacher_name = prepare_args['teacher_name']
    overwrite = prepare_args['overwrite']

    train_cache_path = cache_dir / f'text-cache-train-{teacher_name.replace("/", "-")}.npz'
    val_cache_path = cache_dir / f'text-cache-val-{teacher_name.replace("/", "-")}.npz'

    if overwrite or not train_cache_path.exists():
        logging.info('重写/不存在 Train data 缓存文件，开始处理文件')
        raw_text = []
        coco2017_file = raw_data_dir / 'mscoco' / 'annotations' / 'captions_train2017.json'
        cc_file = raw_data_dir / 'cc' / 'train_cc3m.tsv'
        logging.info(f'read coco2017 text data: {str(coco2017_file)}')
        with cc_file.open('r', encoding='utf8') as f:
            for content in f.readlines():
                raw_text.append(content.split('\t')[0])
        with coco2017_file.open('r', encoding='utf8') as f:
            res = json.load(f)
            for annotation in res['annotations']:
                raw_text.append(annotation['caption'])

        logging.info('All data: {} Begin tokenizing...'.format(len(raw_text)))
        tokenize_text = []
        for text in tqdm(raw_text):
            tokenize_text.append(tokenize(text, truncate=True).squeeze())

        np.savez(train_cache_path, tokenize_text=tokenize_text)

    if overwrite or not val_cache_path.exists():
        logging.info('重写/不存在 Val data 缓存文件，开始处理文件')
        val_image_file_list_path = raw_data_dir / 'mscoco' / 'val2017'
        path_list = []
        tokenized_sentence = []
        captions = []
        file_dir = raw_data_dir / 'mscoco' / 'annotations' / 'captions_val2017.json'
        with file_dir.open('r', encoding='utf8') as f:
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
                captions.append(caption)
                tokenized_sentence.append(tokenize(caption, truncate=True).squeeze())
                path_list.append(val_image_file_list_path / file_name)
        image_rep = encode_images(path_list, teacher_name)
        np.savez(val_cache_path, captions=captions, tokenized_sentence=tokenized_sentence,
                 path_list=path_list, image_rep=image_rep)
    logging.info('Cache生成完成')


class CombineTextDataset:
    def __init__(self, cache_dir='cache', train=True, teacher_name='ViT-B/32'):
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            self.cache_dir.mkdir()
        self.train = train
        self.teacher_name = teacher_name

        cache_path = self.cache_dir / f'text-cache-train-{self.teacher_name.replace("/", "-")}.npz' \
            if self.train else self.cache_dir / f'text-cache-val-{self.teacher_name.replace("/", "-")}.npz'
        logging.info('加载缓存文件')
        if self.train:
            data = np.load(cache_path)
            self.tokenize_text = data['tokenize_text']
        else:
            data = np.load(cache_path)
            self.sentences = data['tokenized_sentence']
            self.captions = data['captions']
            self.path_list = data['path_list']
            self.image_rep = data['image_rep']
        logging.info('加载完成！')

    def __len__(self):
        if self.train:
            return len(self.tokenize_text)
        else:
            return len(self.path_list)

    def __getitem__(self, idx):
        if self.train:
            return self.tokenize_text[idx]

        return Tensor(self.image_rep[idx]), self.captions[idx], self.sentences[idx]