import numpy as np
from PIL import Image
from tqdm import tqdm

from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
import mindspore.dataset.vision.transforms as transforms
import mindformers.models.clip.clip_tokenizer


IMAGE_DATASET_NAME = ['coco', 'data_256', 'imagenet']
IMAGE_PREFIX = {
    'coco': '0',
    'data_256': 'data_256',
    'imagenet': 'imagenet'
}
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

#对图像和文本分别进行编码
def encode_images(path_list, teacher_name: str):
    from clip import load
    image_encode = []
    device = 'cuda'
    model, preprocess = load(teacher_name, device)
    model.set_train(False)
    model.add_flags_recursive(fp16=False)
    for path in tqdm(path_list):
        image = Image.open(path)
        image = preprocess(image).unsqueeze(0)
        image = Tensor(image)
        image_features = model.encode_image(image)
        image_features = image_features.asnumpy()
        image_encode.append(image_features)
    return np.concatenate(image_encode, axis=0)


def encode_texts(caption_list, teacher_name: str):
    from clip import load #, tokenize
    text_encode = []
    device = 'cuda'
    model, preprocess = load(teacher_name, device)
    model.set_train(False)
    model.add_flags_recursive(fp16=False)
    for caption in tqdm(caption_list):
        with context.set_context(mode=context.GRAPH_MODE):#设置成静态图模式，保证速度与内存
            caption = tokenize(caption)
            caption = Tensor(caption)#编码并转换为tensor
            caption = caption.unsqueeze(0)#mindspore要求模型输入带有批次维度
            text_features = model.encode_text(caption)#文本编码
            text_features = text_features.asnumpy()
            text_encode.append(text_features)
    return np.concatenate(text_encode, axis=0)#拼接