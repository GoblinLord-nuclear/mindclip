from mindspore import nn
from mindspore.ops import operations as P
from mindspore.nn import LogSoftmax
class CrossModalAtnnLoss(nn.Cell):
    def __init__(self):
        super(CrossModalAtnnLoss, self).__init__()
        self.transpose = P.Transpose()
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.kl_div = nn.KLDivLoss(reduction='none')

    def forward(self, stu_hidden, tea_img_hidden, tea_txt_hidden):
        img_max_length = P.Shape()(tea_img_hidden)[-2]

        stu_i2t_attn = stu_hidden[:, :, :img_max_length, img_max_length:]             
        stu_t2i_attn = stu_hidden[:, :, img_max_length:, :img_max_length]
        tea_i2t_attn = tea_img_hidden @ self.transpose(tea_txt_hidden, (-2, -1))
        tea_t2i_attn = tea_txt_hidden @ self.transpose(tea_img_hidden, (-2, -1))

        tmp_loss = self.kl_div(LogSoftmax(stu_i2t_attn.astype('float32'), dim=-1), LogSoftmax(tea_i2t_attn.astype('float32'), dim=-1))
        tmp_loss += self.kl_div(LogSoftmax(stu_t2i_attn.astype('float32'), dim=-1), LogSoftmax(tea_t2i_attn.astype('float32'), dim=-1))

        return self.reduce_mean(self.reduce_sum(tmp_loss, dim=-1))