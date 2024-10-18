import mindspore as ms
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint
from mindspore.train.callback import LossMonitor, TimeMonitor
from transformers.optimization import get_cosine_schedule_with_warmup
import os

# 设置环境变量（如果需要）
# os.environ['JSONARGPARSE_DEBUG'] = 'true'
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

# 自定义模型训练和优化器设置
class MyMindSporeTraining:
    def __init__(self, model, train_dataset, val_dataset, epochs=10, batch_size=32, lr=0.001, warmup_steps=1000):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.warmup_steps = warmup_steps

    def configure_optimizers(self):
        optimizer = nn.Adam(self.model.trainable_params(), learning_rate=self.lr)
        lr_scheduler = nn.CosineDecayLR(min_lr=0.0001, max_lr=self.lr, decay_steps=self.epochs * len(self.train_dataset))
        return optimizer, lr_scheduler

    def train(self):
        # 配置优化器和学习率调度器
        optimizer, lr_scheduler = self.configure_optimizers()
        
        # 创建 MindSpore 模型对象
        model_train = Model(self.model, optimizer=optimizer, metrics={'Accuracy': nn.Accuracy()})

        # 配置回调函数
        loss_monitor = LossMonitor(per_print_times=100)
        time_monitor = TimeMonitor(data_size=self.train_dataset.get_dataset_size())
        ckpt_config = CheckpointConfig(save_checkpoint_steps=200, keep_checkpoint_max=5)
        ckpt_callback = ModelCheckpoint(prefix="mymodel", config=ckpt_config)

        # 开始训练
        model_train.train(self.epochs, self.train_dataset, callbacks=[loss_monitor, time_monitor, ckpt_callback], dataset_sink_mode=False)

        # 验证模型
        val_result = model_train.eval(self.val_dataset, dataset_sink_mode=False)
        print(f"Validation Result: {val_result}")


if __name__ == '__main__':
    # 数据集和模型的占位符，实际使用时替换为你的模型和数据集
    train_dataset = ds.ImageFolderDataset("/path/to/train/data").batch(32)
    val_dataset = ds.ImageFolderDataset("/path/to/val/data").batch(32)

    # 初始化你的模型 (占位符，替换为实际模型)
    model = nn.SequentialCell([nn.Dense(512, 256), nn.ReLU(), nn.Dense(256, 10)])

    # 初始化训练类并开始训练
    trainer = MyMindSporeTraining(model, train_dataset, val_dataset, epochs=10, lr=0.001)
    trainer.train()
