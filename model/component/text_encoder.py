from mindspore import nn
import mindspore
from mindspore import load_param_into_net
from mindspore import ops as ops
from mindspore.common.initializer import Normal, initializer
from model.component._common import Transformer, LayerNorm
from model.component.output import ControlOutput, TransformerOutput, TextTransformerOutput
class TextEncoder(nn.Cell):
    def __init__(self, transformer_width, transformer_layers, transformer_heads, context_length, need_layers,
                 vocab_size, embed_dim, tea_transformer_width=None, is_student=True, drop_out=0.,
                 compression_embedding=False, embedding_compression_dim=256):
        super().__init__()
        self.context_length = context_length
        self.transformer_width = transformer_width
        self.transformer_heads = transformer_heads
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.layers = transformer_layers
        
        if compression_embedding:
            self.token_embedding = nn.SequentialCell(*[
                nn.Embedding(vocab_size, embedding_compression_dim),
                nn.Dense(embedding_compression_dim, embed_dim)]
            )
        else:
            self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = mindspore.Parameter(mindspore.numpy.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = mindspore.Parameter(mindspore.numpy.empty(transformer_width, embed_dim))
        self.transformer_para = dict(
            width=self.transformer_width,
            layers=self.layers,
            heads=self.transformer_heads,
            drop_out=drop_out,
            attn_mask=self.build_attention_mask(),
            need_layers=need_layers
        )
        self.transformer = Transformer(**self.transformer_para)

        self.embedding_projection = None
        self.hidden_projection = None
        self.is_student = is_student
        self.no_trans = False
        if transformer_layers == tea_transformer_width:
            self.no_trans = True
        if is_student:
            self.embedding_projection = nn.Dense(transformer_width, tea_transformer_width)
            self.hidden_projection = nn.Dense(transformer_width, tea_transformer_width)
        self.initialize_parameters()
    @property
    def need_layers(self):
        return self.transformer_para['need_layers']
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask=ops.fill(mindspore.float32,(self.context_length, self.context_length),float("-inf"),)#注意这里的type
        ops.triu(mask,diagonal=1)  # zero out the lower diagonal
        return mask
    def encode_text(self, text, control_output: ControlOutput = None):
        if control_output is None:
            control_output = ControlOutput()
        embedding = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = embedding + self.positional_embedding
        embedding_res = x
        transformer_output: TransformerOutput = self.transformer(x, control_output)
        x = self.ln_final(transformer_output.last_layer_output)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        last_layer_output = x @ self.text_projection
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        if self.is_student and not self.no_trans:
            if control_output.need_rep:
                transformer_output.representations = [self.hidden_projection(layer_rep) for layer_rep in
                                                      transformer_output.representations]
            if control_output.need_emb:
                embedding_res = self.embedding_projection(embedding_res)
        if control_output.need_attn_score:
            transformer_output.attention_scores = [ops.where(attn_score == float('-inf'),
                                                               ops.zeros_like(attn_score),
                                                               attn_score) for attn_score in
                                                   transformer_output.attention_scores]
        return TextTransformerOutput(last_representation=last_layer_output[ops.arange(x.shape[0]), ops.argmax(text,dim=-1)],
                                     last_layer_output=last_layer_output,
                                     attention_scores=transformer_output.attention_scores,
                                     attention_probs=transformer_output.attention_probs,
                                     representations=transformer_output.representations,
                                     value_map=transformer_output.value_map,
                                     embedding=embedding_res)
    def initialize_parameters(self):
        self.token_embedding.weight=initializer(Normal(sigma=0.02,mean=0),self.token_embedding.weight.shape,dtype=mindspore.float32)
        self.positional_embedding=initializer(Normal(sigma=0.01,mean=0),self.positional_embedding.shape,dtype=mindspore.float32)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            block.attn.in_proj_weight=initializer(Normal(sigma=attn_std, mean=0.0),block.attn.in_proj_weight.shape,dtype=mindspore.float32)
            block.attn.in_proj_bias=initializer(Normal(sigma=attn_std, mean=0.0),block.attn.in_proj_bias.shape,dtype=mindspore.float32)
            block.attn.out_proj.weight=initializer(Normal(sigma=proj_std, mean=0.0),block.attn.out_proj.weight.shape,dtype=mindspore.float32)
            block.mlp.c_fc.weight=initializer(Normal(sigma=fc_std, mean=0.0),block.mlp.c_fc.weight.shape,dtype=mindspore.float32)
            block.mlp.c_proj.weight=initializer(Normal(sigma=proj_std, mean=0.0),block.mlp.c_proj.weight.shape,dtype=mindspore.float32)
        if self.text_projection is not None:
            self.text_projection=initializer(Normal(sigma=self.transformer.width ** -0.5, mean=0.0),self.text_projection.shape,dtype=mindspore.float32)
    def construct(self, text, control_output: ControlOutput):
        return self.encode_text(text, control_output)    
    def hyper_para(self):
        return {
            'context_length': self.context_length,
            'transformer_width': self.transformer_width,
            'transformer_layers': self.layers,
            'transformer_heads': self.transformer_heads,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
        }
    def init_layers_with_teacher(self, layer_map, teacher_state_dict=None, init_type=None):
        if init_type is None:
            return
        import re
        pattern = re.compile('transformer.resblocks.([\d])')
        stu_layer_num = layer_map.stu_total_layer_num
        tea_layer_num = layer_map.tea_total_layer_num
        tea_state_dict = teacher_state_dict
        my_model_state_dict = self.network.parameters_dict()

        if init_type == 'begin':
            map_layer = lambda x: str(x)
        elif init_type == 'end':
            map_layer = lambda x: str(tea_layer_num - stu_layer_num + x)
        elif init_type == 'mid':
            map_layer = lambda x: str(x * layer_map.step)
        else:
            raise ValueError('the init_type should be begin, end, and mid, but got {}'.format(self.init_type))
        for key in my_model_state_dict.keys():#这里的代码可能是错的
            if key not in tea_state_dict:
                continue
            res = re.findall(pattern, key)
            if not res and not key.startswith('visual'):
                my_model_state_dict[key] = tea_state_dict[key]
            else:
                tea_key = re.sub(re.compile('\d'), map_layer(int(res[0])), string=key, count=1)
                my_model_state_dict[key] = tea_state_dict[tea_key]
                
        load_param_into_net(self,my_model_state_dict)
        print('init with teacher weight success!')