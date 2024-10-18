from mindspore import nn
class LogitsMSE(nn.Cell):
    def __init__(self):
        super(LogitsMSE, self).__init__()
        self.loss = nn.MSELoss()

    def construct(self, stu_logits, tea_logits):
        return self.loss(stu_logits, tea_logits)
