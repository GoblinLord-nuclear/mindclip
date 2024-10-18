from mindspore import nn
from mindspore.ops import operations as P

class AttentionScoreMSE(nn.Cell):
    def __init__(self):
        super(AttentionScoreMSE, self).__init__()
        self.loss = nn.MSELoss()
        self.reduce_sum = P.ReduceSum()

    def forward(self, stu_attn_score, tea_attn_score):
        res_loss = 0
        for layer_num, (stu_out, tea_out) in enumerate(zip(stu_attn_score, tea_attn_score)):
            stu_head_num = stu_out.shape[1]
            tea_head_num = tea_out.shape[1]
            stu_mean_head_out = self.reduce_sum(stu_out, (1,)) / stu_head_num
            tea_mean_head_out = self.reduce_sum(tea_out, (1,)) / tea_head_num
            if layer_num == 0:
                res_loss = self.loss(stu_mean_head_out, tea_mean_head_out)
            else:
                res_loss += self.loss(stu_mean_head_out, tea_mean_head_out)
        res_loss /= len(stu_attn_score)
        return res_loss
