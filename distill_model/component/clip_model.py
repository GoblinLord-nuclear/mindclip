import mindspore; mindspore.set_context(mode=0, device_id=0)
from mindformers import CLIPModel, CLIPConfig

CLIPModel.show_support_list()
# 输出：
# - support list of CLIPModel is:
# -    ['clip_vit_b_32', 'clip_vit_B_16', 'clip_vit_l_14', 'clip_vit_l_14@336']
# - -------------------------------------

# 模型标志加载模型
model = CLIPModel.from_pretrained("clip_vit_b_32")

#模型配置加载模型
config = CLIPConfig.from_pretrained("clip_vit_b_32")
# {'text_config': {'hidden_size': 512, 'vocab_size': 49408, 'max_position_embeddings': 77,
# 'num_hidden_layers': 12}, 'vision_config': {'hidden_size': 768, 'image_size': 224, 'patch_size': 32,
# 'num_hidden_layers': 12}, 'projection_dim': 512, 'ratio': 64, 'checkpoint_name_or_path': 'clip_vit_b_32',
# 'dtype': 'float16'}
model = CLIPModel(config)