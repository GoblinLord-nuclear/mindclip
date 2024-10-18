import mindspore; mindspore.set_context(mode=0, device_id=0)
from mindformers import CLIPModel, CLIPConfig, Transformer

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
if __name__ == '__main__':
    trans_para = {
        'embed_dim': 512,
        'context_length': 77,
        'vocab_size': 49408,
        'transformer_width': 512,
        'transformer_heads': 8,
        'transformer_layers': 4,
    }
    vit_para = {
        'input_resolution': 224,
        'patch_size': 32,
        'width': 768,
        'layers': 4,
        'heads': 12,
        'output_dim': 512,
        'drop_out': 0.1
    }
    m = CLIPModel(False, vit_para, trans_para)
    print(m.state_dict())