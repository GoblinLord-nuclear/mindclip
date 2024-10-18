from dataclasses import dataclass
from typing import List

import mindspore
from mindspore import Tensor


@dataclass
class ControlOutput:
    need_emb: bool = False
    need_attn_score: bool = False
    need_value_map: bool = False
    need_attn_prob: bool = False
    need_rep: bool = False


@dataclass
class VisionTransformerOutput:
    last_representation: Tensor = None
    last_layer_output: Tensor = None
    attention_scores: List[Tensor] = None
    attention_probs: List[Tensor] = None
    representations: List[Tensor] = None
    value_map: Tensor = None,
    embedding: Tensor = None,


@dataclass
class TextTransformerOutput:
    last_representation: Tensor = None
    last_layer_output: Tensor = None,
    attention_scores: List[Tensor] = None
    attention_probs: List[Tensor] = None
    representations: List[Tensor] = None
    value_map: Tensor = None,
    embedding: Tensor = None,


@dataclass
class AttentionOutput:
    attention_output: Tensor = None
    attention_scores: Tensor = None,
    attention_probs: Tensor = None,
    value_map: Tensor = None


@dataclass
class TransformerOutput:
    last_layer_output: Tensor = None
    attention_scores: List[Tensor] = None
    attention_probs: List[Tensor] = None
    representations: List[Tensor] = None
    value_map: Tensor = None


@dataclass
class TransformerLayerOutput:
    hidden_representation: Tensor = None
    attention_scores: Tensor = None
    attention_probs: Tensor = None
    value_map: Tensor = None


@dataclass
class CLIPOutput:
    visual_output: VisionTransformerOutput = None
    text_output: TextTransformerOutput = None
    i2t_logits:Tensor = None
    t2i_logits: Tensor = None


@dataclass
class ResnetOutput:
    last_representation: Tensor = None

@dataclass
class ViltOutput:
    total_output: TransformerLayerOutput = None
    visual_output: Tensor = None
    text_output: Tensor = None
    i2t_logits: Tensor = None
    t2i_logits: Tensor = None
