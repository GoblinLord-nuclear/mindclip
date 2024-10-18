import mindspore; mindspore.set_context(mode=0, device_id=0)
from mindformers.trainer import Trainer
from mindformers.tools.image_tools import load_image
# 初始化预训练任务
trainer = Trainer(task='contrastive_language_image_pretrain',
    model='clip_vit_b_32',
    train_dataset='Flickr8k')
trainer.train() # 开启预训练

#初始化零样本图像分类下游任务
trainer = Trainer(task='zero_shot_image_classification',
    model='clip_vit_b_32',
    eval_dataset='D:/BaiduNetdiskDownload/cifar-100-python')  
img = load_image("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")

# 方式1: 使用训练好的权重进行评估和推理
trainer.evaluate(eval_checkpoint=True)
predict_result = trainer.predict(predict_checkpoint=True, input_data=img, top_k=3)
print(predict_result)

# 方式2: 从obs下载训练好的权重并进行评估和推理
trainer.evaluate()  #下载权重进行评估
predict_result = trainer.predict(input_data=img, top_k=3)  #下载权重进行推理
print(predict_result)