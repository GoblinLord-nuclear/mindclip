from mindspore import nn
import mindspore.ops as ops
class OutKLLoss(nn.Cell):
    def __init__(self, t):
        super().__init__()
        self.loss = nn.KLDivLoss(reduction='sum')

        self.temperature = t

    def construct(self, stu_out, tea_out):
        return self.loss(
            ops.log_softmax(stu_out / self.temperature, axis=1),
            ops.softmax(tea_out / self.temperature, axis=1)
        ) * self.temperature ** 2
