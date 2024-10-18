import math
from collections import OrderedDict
from typing import List, Optional

import mindspore
import mindspore.ops as ops
from mindspore import nn
from mindspore import Parameter
from mindspore import Tensor
from mindspore.common.initializer import initializer, XavierUniform
from .output import ControlOutput, AttentionOutput, TransformerOutput, VisionTransformerOutput, \
    TransformerLayerOutput
from mindspore import dtype as mstype
import mindspore.numpy as mnp
class LayerNorm(nn.LayerNorm):
    """
    Subclass MindSpore's LayerNorm to handle fp16 by casting to fp32 internally.
    """
    def __init__(self, normalized_shape, epsilon=1e-7):
        super(LayerNorm, self).__init__(normalized_shape, epsilon=epsilon, begin_norm_axis=-1, begin_params_axis=-1)
        self.cast = ops.Cast()

    def construct(self, x):
        orig_type = x.dtype
        x = self.cast(x, mstype.float32)  # Cast to fp32
        output = super().construct(x)
        return self.cast(output, orig_type)  # Cast back to original dtype



class QuickGELU(nn.Cell):
    def construct(self, x: Tensor):
        return x * ops.sigmoid(1.702 * x)


class MultiheadAttention(nn.Cell):
    def __init__(self, hidden_size, num_attention_heads, drop_prob):
        super(MultiheadAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.in_proj_weight = Parameter(initializer(XavierUniform(), (3 * hidden_size, hidden_size), mindspore.float32))
        self.in_proj_bias = Parameter(initializer('zeros', (3 * hidden_size,), mindspore.float32))

        self.out_proj = nn.Dense(hidden_size, hidden_size)

        self.dropout = nn.Dropout(1.0 - drop_prob)

        self.matmul_trans_b = ops.BatchMatMul(transpose_b=True)
        self.softmax = nn.Softmax(axis=-1)
        self.reshape = ops.Reshape()

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = ops.reshape(x, new_x_shape)
        x = ops.transpose(x, (0, 2, 1, 3))
        return x

class MultiHeadAttention(nn.Cell):
    def __init__(self, num_attention_heads, attention_head_size, all_head_size, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.all_head_size = all_head_size
        self.dropout = nn.Dropout(1 - dropout_rate)
        
        # 定义输入投影的权重和偏置
        self.in_proj_weight = Parameter(initializer('xavier_uniform_', [all_head_size * 3, all_head_size]), name='in_proj_weight')
        self.in_proj_bias = Parameter(initializer('zeros', [all_head_size * 3]), name='in_proj_bias')
        
        # 输出投影
        self.out_proj = nn.Dense(all_head_size, all_head_size)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = ops.reshape(x, new_x_shape)
        return ops.transpose(x, (0, 2, 1, 3))

    def construct(self, hidden_states, control_output: ControlOutput, attention_mask=None):
        mixed_x_layer = ops.matmul(hidden_states, ops.transpose(self.in_proj_weight, (1, 0))) + self.in_proj_bias
        q, k, v = ops.split(mixed_x_layer, 3, -1)
        
        query_layer = self.transpose_for_scores(q)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)

        value_map = None
        return_attention_scores = None
        return_attention_probs = None
        
        if control_output.need_value_map:
            value_map = ops.matmul(value_layer, ops.transpose(value_layer, (0, 1, 3, 2)))
            value_map = value_map / math.sqrt(self.attention_head_size)
            value_map = nn.Softmax(axis=-1)(value_map)

        attention_scores = ops.matmul(query_layer, ops.transpose(key_layer, (0, 1, 3, 2)))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(axis=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = ops.matmul(attention_probs, value_layer)

        context_layer = ops.transpose(context_layer, (0, 2, 1, 3))
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = ops.reshape(context_layer, new_context_layer_shape)
        context_layer = self.out_proj(context_layer)

        if control_output.need_attn_prob:
            return_attention_probs = attention_probs
        if control_output.need_attn_score:
            return_attention_scores = attention_scores
        
        return AttentionOutput(context_layer, return_attention_scores, return_attention_probs, value_map)
class ResidualAttentionBlock(nn.Cell):
    def __init__(self, d_model: int, n_head: int, attn_mask: Tensor = None, drop_prob: float = 0.1):
        super(ResidualAttentionBlock, self).__init__()

        self.attn = MultiheadAttention(d_model, n_head, dropout_prob=drop_prob)
        self.ln_1 = nn.LayerNorm((d_model,), epsilon=1e-12)
        self.mlp = nn.SequentialCell([
            ("c_fc", nn.Dense(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Dense(d_model * 4, d_model))
        ])
        self.ln_2 = nn.LayerNorm((d_model,), epsilon=1e-12)
        self.attn_mask = Parameter(attn_mask, name='attn_mask', requires_grad=False) if attn_mask is not None else None
        self.cast = ops.Cast()

    def attention(self, x: Tensor, control_output: ControlOutput):
        if self.attn_mask is not None:
            self.attn_mask = self.cast(self.attn_mask, ops.dtype(x))
        return self.attn(x, attention_mask=self.attn_mask, control_output=control_output)

    def construct(self, x: Tensor, control_output: ControlOutput):
        attention_output: AttentionOutput = self.attention(self.ln_1(x), control_output)
        x = x + attention_output.attention_output
        x = x + self.mlp(self.ln_2(x))
        return TransformerLayerOutput(x, attention_output.attention_scores,
                                      attention_output.attention_probs, attention_output.value_map)

# Please note that Mindspore uses construct method instead of forward method used in PyTorch.


class Transformer(nn.Cell):
    def __init__(self, width: int, layers: int, heads: int,
                 need_layers: Optional[List[int]],
                 attn_mask: Tensor = None, drop_out: float = 0.1):
        super(Transformer, self).__init__()
        self.width = width
        self.layers = layers
        need_layers = need_layers if need_layers is not None else [i for i in range(layers)]
        self.need_layers = need_layers
        self.resblocks = nn.CellList(
            [ResidualAttentionBlock(width, heads, attn_mask, drop_out) for _ in range(layers)])

    def construct(self, x: Tensor, control_output: ControlOutput):
        """
        calculate the Transformer layers
        :param control_output: Type: ControlOutput, For Control the Transformer output
        :param x: hidden state input
        :return: a TransformerOutput(hidden state, attention_scores, attention_probs, representations, value_map)
        """
        value_map = None
        attention_scores = []
        representations = []
        attention_probs = []

        for i, layer in enumerate(self.resblocks):
            layers_output: TransformerLayerOutput = layer(x, control_output)
            x = layers_output.hidden_representation
            if i not in self.need_layers:
                continue
            if control_output.need_rep:
                representations.append(layers_output.hidden_representation)
            if control_output.need_attn_score:
                attention_scores.append(layers_output.attention_scores)
            if control_output.need_attn_prob:
                attention_probs.append(layers_output.attention_probs)
            value_map = layers_output.value_map
        return TransformerOutput(x, attention_scores, attention_probs, representations, value_map)


class VisionTransformer(nn.Cell):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 need_layers: Optional[List], drop_out: float = 0):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = Parameter(Tensor(scale * mnp.random.randn(width)))
        self.positional_embedding = Parameter(Tensor(scale * mnp.random.randn((input_resolution // patch_size) ** 2 + 1, width)))
        self.ln_pre = nn.LayerNorm((width,))

        self.transformer = Transformer(width, layers, heads, drop_out=drop_out, need_layers=need_layers)

        self.ln_post = nn.LayerNorm((width,))
        self.proj = Parameter(Tensor(scale * mnp.random.randn(width, output_dim)))

    def construct(self, x, control_output):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.transpose(0, 2, 1)  # shape = [*, grid ** 2, width]
        class_embedding = self.class_embedding.expand_dims(0).broadcast_as((x.shape[0], 1, x.shape[-1]))
        x = ops.Concat(1)([class_embedding, x])  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding

        embeddings = None
        if control_output.need_emb:
            embeddings = x

        x = self.ln_pre(x)
        vision_output: TransformerOutput = self.transformer(x, control_output)
        x = self.ln_post(vision_output.last_layer_output)

        if self.proj is not None:
            x = x @ self.proj

        return VisionTransformerOutput(last_representation=x[:, 0, :],
                                       last_layer_output=x,
                                       attention_scores=vision_output.attention_scores,
                                       attention_probs=vision_output.attention_probs,
                                       representations=vision_output.representations,
                                       value_map=vision_output.value_map,
                                       embedding=embeddings)