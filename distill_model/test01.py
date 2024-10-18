import mindspore
import numpy as np
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.ops import operations as P
from mindspore.dataset.vision import Inter
from mindspore.dataset.transforms.c_transforms import TypeCast
from mindspore.dataset.vision.c_transforms import Decode
import mindspore.dataset as ds

import json
import mindspore.dataset.vision.transforms as C
from mindspore import Tensor
import dataset.data.text_image_datamodule as D
import mindspore.dataset.vision as CV
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

#context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
dataset_dir = "D:/Download/DistillCLIP/coco2017/train2017/train2017"
annotation_file = "D:/Download/DistillCLIP/coco2017/annotations_trainval2017/captions_train2017.json"

dataset_m=D.TextImageDataModule(image_path="D:/Download/DistillCLIP/coco2017/train2017/train2017",
                                batch_size=64,
                                workers=4)
transform=dataset_m.make_transform(is_train=True)

coco_dataset = ds.CocoDataset(dataset_dir=dataset_dir,
                              annotation_file=annotation_file,
                              task='Captioning',
                              num_samples=None,
                              num_parallel_workers=4,  # 设置线程数
                              shuffle=True,  # 做混洗
                              decode=True,  # 不decode无法归一化
                              cache=None,
                              extra_metadata=False,
                              decrypt=None)


from distrill_model.model.model_distill import DistillModel
#loss_control_para=
#download=
clip_model = DistillModel(download_root=download_root,
                          teacher_name='clip',
                          student_encoder=torch.nn.module,
                          loss_control_para=loss_comtrol_para)

def evaluate(model, dataset):
    model.set_train(False)
    for data in dataset.create_tuple_iterator():
        images = Tensor(data[0].astype(np.float32))
        outputs = model(images)

evaluate(clip_model, coco_dataset)
