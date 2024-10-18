import os
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
# 设置路径
ckpt_path = "./checkpoints/"
ckpt_name = "checkpoint"

config_ck = CheckpointConfig(save_checkpoint_steps=1000, keep_checkpoint_max=5)

ckpoint_cb = ModelCheckpoint(prefix=ckpt_name, directory=ckpt_path, config=config_ck)

save_dir = "./checkpoint"
os.makedirs(save_dir, exist_ok=True)

# 定义保存模型的回调函数
ckpt_config = CheckpointConfig(save_checkpoint_steps=500, keep_checkpoint_max=10)
ckpoint_cb = ModelCheckpoint(prefix="lenet_ckpt", directory=save_dir, config=ckpt_config)