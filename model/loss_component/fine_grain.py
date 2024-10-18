import mindspore
from mindspore import nn
import mindspore.ops as f
from mindspore import Tensor
class FineGrainLoss(nn.Cell):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction='mean')
        self.argmax=f.ArgMaxWithValue(axis=-1)
        self.stack=f.Stack(axis=0)
    def construct(self, image_out, text_out):
        def cal_similarity(query_features, respond_features):
            res=[]
            for q in query_features:
                similarity=f.MatMul(q, f.Transpose(respond_features, (0, 2, 1)))
                max_res=self.argmax(similarity)
                mean_res = f.ReduceMean(max_res,-1)
                res.append(mean_res)
                similarity_total = self.stack(res)
                return similarity_total
        i2t_similarity = cal_similarity(image_out, text_out)
        t2i_similarity = cal_similarity(text_out, image_out)
        label = f.arange(i2t_similarity.shape[0], device=i2t_similarity.device)
        loss_i2t = self.loss(i2t_similarity, label)
        loss_t2i = self.loss(t2i_similarity, label)
        
        return (loss_i2t + loss_t2i) / 2