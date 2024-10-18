import mindspore.dataset as ds
import json
import mindspore.dataset.vision.transforms as C
from mindspore import Tensor
import data.text_image_datamodule as D
import mindspore.dataset.vision as CV
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

dataset_dir = "D:/Download/DistillCLIP/coco2017/train2017/train2017"
annotation_file = "D:/Download/DistillCLIP/coco2017/annotations_trainval2017/captions_train2017.json"

resize_height, resize_width = 224, 224
rescale = 1.0 / 255.0
shift = 0.0
pad_mode = "pad"
padding = 0
dataset_m=D.TextImageDataModule(image_path="D:/Download/DistillCLIP/coco2017/train2017/train2017",
                                batch_size=64,
                                workers=4)
transform=dataset_m.make_transform(is_train=True)
#random_crop = C.RandomCropDecodeResize((resize_height, resize_width))
#random_crop=D.TextImageDataModule.make_transform(Ture)
#random_horizontal_flip = C.RandomHorizontalFlip()
#normalize = C.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                        #std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
#changeswap = C.HWC2CHW()

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
#coco_dataset=coco_dataset.batch(5)
#batched_image, batched_label = next(iter(coco_dataset))

#print("batched_image.shape:",batched_image.shape,"batched_label.shape:",batched_label.shape)
#coco_dataset=coco_dataset.take(3)
#for i,(image,annotation) in enumerate(coco_dataset.create_dict_iterator()):
#    if i>=5:
#        break
#    print(f"Take 3 batches, {i + 1}/3 batch:", image.shape, annotation.shape)

for idx, data in enumerate(coco_dataset.create_dict_iterator()):
    if idx>=5:
        break
    print(f"Sample {idx + 1}:")
    image_data = data["image"]
    print("Keys in data:", data.keys())
    print("Image shape:", image_data.shape)
    print("Image type:", image_data.dtype)
    caption_data=data["captions"]
    print("caption shape",caption_data.shape)
    print("caption type:",caption_data.dtype)
print("test finished")
 # 预处理和增强
transform_test=[
            CV.Resize(224),
            CV.CenterCrop(224),
            #CV.ToTensor(),
            CV.Normalize(IMAGE_MEAN,IMAGE_STD),
            CV.HWC2CHW()
        ]
coco_dataset = coco_dataset.map(operations=transform,
                           input_columns=["image"],
                           num_parallel_workers=4)

print("------------------------------")


for idx, data in enumerate(coco_dataset.create_dict_iterator()):
    if idx>=5:
        break
    print(f"Sample {idx + 1}:")
    image_data = data["image"]
    print("Keys in data:", data.keys())
    print("Image shape:", image_data.shape)
    print("Image type:", image_data.dtype)
    caption_data=data["captions"]
    print("caption shape",caption_data.shape)
    print("caption type:",caption_data.dtype)
    #print("BBox:", data["bbox"])
    #print("Category ID:", data["category_id"])
    #print("Is crowd:", data["iscrowd"])
    #print("annotation:",data["annotation"].shape)
#    image_id=find_target(data["bbox"],data["category_id"],data["iscrowd"])
#    print("text shape:",find_text(image_id).caption.shape)
    print()
print("OK了")
