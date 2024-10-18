from mindspore import nn
class OutL1Loss(nn.Cell):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def construct(self, stu_out, tea_out):
        return self.loss(stu_out, tea_out)