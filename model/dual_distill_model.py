from typing import *
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn, ops
from mindspore.train import Model, Callback, CheckpointConfig, ModelCheckpoint
from mindspore.train.callback import LossMonitor, TimeMonitor

from ._loss import LossCalculator
from ._metrics import cal_flop, cal_speed
from .utils import teacher_load
from .component.clip_model import CLIPModel
from .component.image_encoder import ImageEncoder
from .component.output import CLIPOutput
from .component.weight_share_model import RepeatVisionTransformer

def load_weight(image_student, text_student, load_path): # 蒸馏时候老师跟学生都存下来的
    def load_one_model(model: nn.Cell, cpk: Optional[str]):
        if cpk is None:
            raise ValueError('the cpk is None! if you set the load_path parameter in model,'
                             ' you should give the image and text checkpoint path')
        save_res = ms.load_checkpoint(cpk)
        state_dict = {
            k.replace('student.', ''): v
            for k, v in save_res.items() if k.startswith('student')
        }

        ms.load_param_into_net(model, state_dict)
        return model

    image_student = load_one_model(image_student, load_path['image'])
    text_student = load_one_model(text_student, load_path['text'])
    return image_student, text_student

class DualDistillModel(nn.Cell):
    def __init__(self, image_student: nn.Cell, text_student: nn.Cell,
                 loss_control_para: Dict, warm_steps, total_steps, weight_decay, lr: float,
                 download_root: str, norm=False, teacher_name: str = 'ViT-B/32', freeze_embed: bool = False,
                 unfreeze_epoch: int = None, load_path: Dict = None, teacher_need_layers: List = None,
                 freeze_prefix: List = None):
        super(DualDistillModel, self).__init__()

        if load_path:
            image_student, text_student = load_weight(image_student, text_student, load_path)
        self.student = CLIPModel(True, image_student, text_student, norm)

        self.teacher = teacher_load(teacher_name, download_root, 'all', need_layers=teacher_need_layers)
        for p in self.teacher.get_parameters():
            p.requires_grad = False

        self.loss_control = LossCalculator(**loss_control_para)
        self.need_return_para = self.loss_control.get_control_output() # 根据loss进行输出控制
        if freeze_embed:
            self.freeze_image_embedding()
        self.unfreeze_epoch = unfreeze_epoch

        self.freeze_with_prefix(prefix_list=freeze_prefix)
        # define acc top k
        self.k_list = [i for i in [1, 3, 5]]

    def construct(self, image, text):
        student_outs: CLIPOutput = self.student(text, image, self.need_return_para)
        teacher_outs: CLIPOutput = self.teacher(text, image, self.need_return_para)
        if self.norm:
            student_outs, teacher_outs = norm_last_representation(student_outs, teacher_outs)
        return student_outs, teacher_outs

    def train_step(self, inputs):
        self.teacher.set_train(False)
        student_outs, teacher_outs = self.construct(*inputs)
        loss, cal_res = self.loss_control(student_outs, teacher_outs, 'all')
        return loss, cal_res

    def val_step(self, inputs):
        student_outs, teacher_outs = self.construct(*inputs)
        loss, cal_res = self.loss_control(student_outs, teacher_outs, 'all')
        return loss, cal_res

    def unfreeze_embed(self):
        for p in self.student.get_parameters():
            p.requires_grad = True

    def freeze_with_prefix(self, prefix_list):
        if prefix_list is None:
            return

        for p in self.student.get_parameters():
            for prefix in prefix_list:
                if p.name.startswith(prefix):
                    print(f'freeze {p.name}')
                    p.requires_grad = False

    def freeze_image_embedding(self):
        freeze_key = ['visual.conv1.weight', 'visual.class_embedding', 'visual.positional_embedding']
        teacher_keys = ['image_encoder.' + k for k in freeze_key]
        if isinstance(self.student.image_encoder, RepeatVisionTransformer):
            student_weights = self.student.parameters_dict()
            base_key = ['patch_embed.proj.weight', 'cls_token', 'pos_embed']
            student_keys = ['image_encoder.' + k for k in base_key]

            for s_k, t_k in zip(student_keys, teacher_keys):
                weights = self.teacher.parameters_dict()[t_k]
                student_weights[s_k].set_data(weights)

            self.student.set_parameters(student_weights)
            for p in self.student.get_parameters():
                if p.name in student_keys:
                    p.requires_grad = False
        elif isinstance(self.student.image_encoder, ImageEncoder):
            student_keys = teacher_keys
            student_weights = self.student.parameters_dict()
            for k in teacher_keys:
                student_weights[k].set_data(self.teacher.parameters_dict()[k])
            self.student.set_parameters(student_weights)
            for p in self.student.get_parameters():
                if p.name in student_keys:
                    p.requires_grad = False


def norm_and_logits(img_encode, text_encode):
    img_encode = img_encode / img_encode.norm(axis=1, keepdims=True)
    text_encode = text_encode / text_encode.norm(axis=1, keepdims=True)
    logits = ops.matmul(img_encode, text_encode.T)
    return logits, logits.T


def norm_last_representation(stu_outs, tea_outs):
    stu_outs.visual_output.last_representation /= stu_outs.visual_output.last_representation.norm(axis=-1, keepdims=True)
    stu_outs.text_output.last_representation /= stu_outs.text_output.last_representation.norm(axis=-1, keepdims=True)
    tea_outs.visual_output.last_representation /= tea_outs.visual_output.last_representation.norm(axis=-1, keepdims=True)
    tea_outs.text_output.last_representation /= tea_outs.text_output.last_representation.norm(axis=-1, keepdims=True)

    return stu_outs, tea_outs

