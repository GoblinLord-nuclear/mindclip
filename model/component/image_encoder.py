import mindspore
from mindspore import nn
from mindspore.common.initializer import Normal, initializer
from ._common import VisionTransformer
from .output import ControlOutput, VisionTransformerOutput
import mindspore.ops as ops
from mindspore import load_param_into_net
class ImageEncoder(nn.Cell):
    def __init__(self, is_student, vit_paras, tea_transformer_width=None):
        super().__init__()
        self.layers = vit_paras['layers']
        if 'need_layers' not in vit_paras or vit_paras['need_layers'] is None:
            vit_paras['need_layers'] = tuple(range(self.layers))
        self.vit_paras = vit_paras
        self.visual = VisionTransformer(**vit_paras) # need layers会帮助收集结果
        self.is_student = is_student
        self.embedding_projection = None
        self.hidden_projection = None

        self.no_trans = False
        if self.vit_paras['width'] == tea_transformer_width:
            self.no_trans = True
        if is_student:
            self.embedding_projection = nn.Dense(vit_paras['width'], tea_transformer_width)
            self.hidden_projection = nn.Dense(vit_paras['width'], tea_transformer_width)
        self.initialize_parameters()

    @property
    def need_layers(self):
        return self.vit_paras['need_layers']

    @property
    def output_layer(self):
        return self.visual.proj

    def initialize_parameters(self):
        self.visual.class_embedding =initializer(Normal(sigma=0.02),self.visual.class_embedding.shape,dtype=mindspore.float32)
        normal_initializer_pos =initializer(Normal(sigma=0.01),normal_initializer_pos.shape,dtype=mindspore.float32)

        proj_std = (self.visual.transformer.width ** -0.5) * ((2 * self.visual.transformer.layers) ** -0.5)
        attn_std = self.visual.transformer.width ** -0.5
        fc_std = (2 * self.visual.transformer.width) ** -0.5
        for block in self.visual.transformer.resblocks:
            block.attn.in_proj_weight=initializer(Normal(sigma=attn_std, mean=0.0),block.attn.in_proj_weight.shape,dtype=mindspore.float32)
            block.attn.in_proj_bias=initializer(Normal(sigma=attn_std, mean=0.0),block.attn.in_proj_bias.shape,dtype=mindspore.float32)
            block.attn.out_proj.weight=initializer(Normal(sigma=proj_std, mean=0.0),block.attn.out_proj.weight.shape,dtype=mindspore.float32)
            block.mlp.c_fc.weight=initializer(Normal(sigma=fc_std, mean=0.0),block.mlp.c_fc.weight.shape,dtype=mindspore.float32)
            block.mlp.c_proj.weight=initializer(Normal(sigma=proj_std, mean=0.0),block.mlp.c_proj.weight.shape,dtype=mindspore.float32)

    def encode_image(self, image, control_output: ControlOutput = None):
        if control_output is None:
            control_output = ControlOutput()
        vit_output: VisionTransformerOutput = self.visual(image, control_output)
        if self.is_student and not self.no_trans:
            if control_output.need_rep:
                vit_output.representations = [self.hidden_projection(layer_rep) for layer_rep in
                                              vit_output.representations]
            if control_output.need_emb:
                vit_output.embedding = self.embedding_projection(vit_output.embedding)
        if control_output.need_attn_score:
            zeros_like = ops.ZerosLike()
            vit_output.attention_scores = [ops.where(attn_score == float('-inf'),
                                                     zeros_like(attn_score),
                                                     attn_score) for attn_score in vit_output.attention_scores]

        return vit_output

    def construct(self, image, control_output: ControlOutput):
        return self.encode_image(image, control_output)

    def init_layers_with_teacher(self, layer_map, teacher_state_dict=None, init_type=None):
        import re
        pattern = re.compile('visual.transformer.resblocks.(\\d)')
        stu_layer_num = layer_map.stu_total_layer_num
        tea_layer_num = layer_map.tea_total_layer_num
        tea_state_dict = teacher_state_dict
        my_model_state_dict = self.visual.state_dict()
        if init_type is None:
            return
        elif init_type == 'begin':
            map_layer = lambda x: str(x)
        elif init_type == 'end':
            map_layer = lambda x: str(tea_layer_num - stu_layer_num + x)
        elif init_type == 'mid':
            map_layer = lambda x: str(x * layer_map.step)
        else:
            raise ValueError('the init_type should be begin, end, and mid, but got {}'.format(self.init_type))

        for key in my_model_state_dict:
            res = re.findall(pattern, key)
            if key not in tea_state_dict:
                continue
            if not res:
                my_model_state_dict[key] = tea_state_dict[key]
            else:
                tea_key = re.sub(re.compile('\\d'), map_layer(int(res[0])), string=key, count=1)
                my_model_state_dict[key] = tea_state_dict[tea_key]
        load_param_into_net(self.visual,my_model_state_dict)
        print('init with teacher weight success!')

    def hyper_para(self):
       
        return self.vit_paras