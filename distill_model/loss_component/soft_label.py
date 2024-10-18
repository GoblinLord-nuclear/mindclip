from mindspore import nn
import mindspore.ops as f


class SoftLabel(nn.Module):
    def __init__(self, temperature):
        super(SoftLabel, self).__init__()
        self.loss = nn.KLDivLoss(reduction='sum')
        self.temperature = temperature

    def construct(self, stu_logits, tea_logits):
        x=f.softmax(stu_logits / self.temperature, axis=1)
        x=f.log(x)
        logits_kl_loss = self.loss(
            x,
            f.softmax(tea_logits / self.temperature, axis=1)
        ) * self.temperature ** 2
        return logits_kl_loss
