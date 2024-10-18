from typing import *

import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore import Tensor, context
#import wandb
from matplotlib import pyplot as plt
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.callback import LossMonitor
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore import context, Tensor
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from mindspore.nn.optim import Adam
#from mindspore.train.lr_scheduler import PolynomialDecayLR
#这里添加了注释
from mindspore.dataset import transforms
from mindspore.dataset.vision import vision_utils
from mindspore.dataset.vision import Inter
from mindspore.train.serialization import save_checkpoint
from mindspore.dataset.vision.utils import to_tensor
from mindspore import ops

from thop import profile

from ._loss import LossCalculator
from ._metrics import cal_flop, cal_speed
from .utils import teacher_load
from .component.image_encoder import ImageEncoder
from .component.weight_share_model import RepeatVisionTransformer


class DistillModel(nn.Cell):
    def __init__(self, student_encoder: nn.Module,
                 teacher_name: str, loss_control_para: Dict, download_root: str, freeze_embed: bool = False,
                 teacher_need_layers: List = None, model_type: str = 'image',
                 warm_steps=10, total_steps=200, weight_decay=1e-3, lr: float = 1e-3, norm: bool = False,
                 unfreeze_epoch=None):
        super().__init__()
        if model_type not in ['text', 'image']:
            raise ValueError(f"the model_type should in ['text', 'image'], bug got {model_type}")
        self.save_hyperparameters(ignore=['student_encoder'])

        self.student = student_encoder
        self.teacher_name = teacher_name
        self.teacher = teacher_load(teacher_name, download_root, model_type, need_layers=teacher_need_layers)
        self.loss_control = LossCalculator(**loss_control_para)
        self.need_return_para = self.loss_control.get_control_output()

        for p in self.teacher.parameters():
            p.requires_grad = False
        if model_type == 'image' and freeze_embed:
            self.freeze_image_embedding()

        self.k_list = [i for i in [1, 3, 5, 10, 20, 50]]

    def on_train_start(self):
        self.logger_begin()

    @rank_zero_only
    def logger_begin(self):
        # WandbLogger的日志记录
        if isinstance(self.logger, WandbLogger):
            self.logger.log_hyperparams({'student_para': self.student.hyper_para()})
            self.logger.experiment.log_code()
            wandb.define_metric(name='val_stu_acc/stu_acc_top1', summary='max')
            wandb.define_metric(name='val_stu_acc/stu_acc_top10', summary='max')
            wandb.define_metric(name='val_stu_acc/stu_acc_top50', summary='max')
        #mindspore没有tensorboard怎么实现？
        elif isinstance(self.logger, TensorBoardLogger):
            self.logger.log_hyperparams(self.hparams, {"hp/stu_acc_top1": 0, "hp/stu_acc_top10": 0})

    # 测试模型推理速度
    def speed_test(self, model, dummy_input, prefix):

        with ops.stop_gradient():
            flops, param = cal_flop(model, dummy_input)
            mean_syn, std_syn, mean_fps = cal_speed(model, dummy_input)
            metric_dict = {
                f'{prefix}_flops': flops,
                f'{prefix}_param': param,
                f'{prefix}_mean_times': mean_syn,
                f'{prefix}_std_times': std_syn,
                f'{prefix}_mean_fps': mean_fps
            }
            self.log_dict(metric_dict, sync_dist=True)

    # 前向传播
    def forward(self, inputs):
        student_outs = self.student(inputs, self.need_return_para)
        with ops.stop_gradient(inputs):
            teacher_outs = self.teacher(inputs, self.need_return_para)
        if self.hparams.norm:
            student_outs.last_representation /= student_outs.last_representation.norm(dim=-1, keepdim=True)
            teacher_outs.last_representation /= teacher_outs.last_representation.norm(dim=-1, keepdim=True)
        return student_outs, teacher_outs

    def on_train_epoch_start(self) -> None:
        if self.hparams.unfreeze_epoch:
            if self.current_epoch >= self.unfreeze_epoch:
                self.unfreeze_embed()
                self.hparams.unfreeze_epoch = False

    def training_step(self, inputs, batch_idx):
        self.teacher.eval()
        student_outs, teacher_outs = self.forward(inputs)
        loss, cal_res = self.loss_control(student_outs, teacher_outs, self.hparams.model_type)
        #cal_res用一个字典代替
        self.log_info('train_loss', loss, cal_res, batch_size=len(inputs))
        return loss

    # 定义一个验证步骤
    def validation_step(self, batch, batch_idx):

        student_outs, teacher_outs = self.forward(inputs)
        loss, cal_res = self.loss_control(student_outs, teacher_outs, self.hparams.model_type)

        stu_logits, tea_logits = norm_and_logits(
            contrary_rep, student_outs.last_representation, teacher_outs.last_representation)[:2]

        self.log_info('val_loss', loss, cal_res, batch_size=len(batch))
        self.log_acc(stu_logits, section='val_step', prefix='stu')
        self.log_acc(tea_logits, section='val_step', prefix='tea')
        self.log_diag_score(stu_logits, section='val_step', prefix='stu')

        return {
            'student': self.all_gather(student_outs.last_representation),
            'teacher': self.all_gather(teacher_outs.last_representation),
            'contrary_rep': self.all_gather(contrary_rep)
        }

    # 验证步骤结束
    def validation_step_end(self, step_out):
        return step_out

    def validation_epoch_end(self, outputs) -> None:
        stu_outs = []
        tea_outs = []
        contrary_reps = []
        for batch in outputs:
            student_out, teacher_out, contrary_rep = batch['student'], batch['teacher'], batch['contrary_rep']
            embedding = student_out.shape[-1]
            stu_outs.append(student_out.reshape(-1, embedding))
            tea_outs.append(teacher_out.reshape(-1, embedding))
            contrary_reps.append(contrary_rep.reshape(-1, embedding))
        stu_outs = ms.ops.concat(stu_outs, dim=0).float()  # dim为多少哪个维度就会变化，这是对整个epoch计算
        tea_outs = ms.ops.concat(tea_outs, dim=0).float()
        contrary_reps = ms.ops.concat(contrary_reps, dim=0).float()
        stu_logits, tea_logits = norm_and_logits(contrary_reps, stu_outs, tea_outs)[:2]

        self.log_acc(stu_logits, section='val_stu_acc', prefix='stu')
        self.log_diag_score(stu_logits, section='val_stu_score', prefix='stu')

        if self.current_epoch == 0:
            self.log_diag_score(tea_logits, section='val_tea_score', prefix='tea')
            self.log_acc(tea_logits, section='val_tea_acc', prefix='tea')
        return

    def log_acc(self, logits, section, prefix):
        label = Tensor(np.arange(logits.shape[0]), dtype=mindspore.int32)
        for k in self.k_list:
            acc = accuracy(logits, label, topk=k, axis=1)
            # 在 MindSpore 中没有直接的日志记录器类似于 PyTorch 中的 WandbLogger 或 TensorBoardLogger
            # 因此，这里省略日志记录相关的内容

    def unfreeze_embed(self):
        for _, p in self.student.cells_and_names():
            if isinstance(p, nn.Parameter) and p.requires_grad:
                p.set_trainable(True)

    def freeze_image_embedding(self):
        student_weights = self.student.parameters_dict()
        if isinstance(self.student, RepeatVisionTransformer):
            stu_keys = ['patch_embed.proj.weight', 'cls_token', 'pos_embed']
            tea_keys = ['visual.conv1.weight', 'visual.class_embedding', 'visual.positional_embedding']
            for s_k, t_k in zip(stu_keys, tea_keys):
                weights = self.teacher.state_dict()[t_k]
                if 'cls_token' in s_k:
                    weights = weights.unsqueeze(0).unsqueeze(0)
                if 'pos_embed' in s_k:
                    weights = weights.unsqueeze(0)
                student_weights[s_k] = Tensor(weights)

            self.student.set_parameters_dict(student_weights)
            for _, p in self.student.cells_and_names():
                if isinstance(p, nn.Parameter) and p.requires_grad:
                    p.set_trainable(False)
        elif isinstance(self.student, ImageEncoder):
            freeze_key = ['visual.conv1.weight', 'visual.class_embedding', 'visual.positional_embedding']
            for k in freeze_key:
                student_weights[k] = self.teacher.state_dict()[k]
            self.student.load_state_dict(student_weights)
            for n, p in self.student.named_parameters():
                if n in freeze_key:
                    p.requires_grad = False

def norm_and_logits(encode, stu_encode, tea_encode):
    encode = encode / encode.norm(dim=1, keepdim=True)
    encode = encode.float()
    stu_encode = stu_encode / stu_encode.norm(dim=1, keepdim=True)
    tea_encode = tea_encode / tea_encode.norm(dim=1, keepdim=True)
    stu_logits = stu_encode @ encode.t()
    tea_logits = tea_encode @ encode.t()
    return stu_logits, tea_logits, stu_logits.T, tea_logits.T