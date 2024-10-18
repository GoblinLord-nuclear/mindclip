from mindspore import nn
from mindspore import ops as f
class LastValueMapKL(nn.Cell):
    def __init__(self):
        super(LastValueMapKL, self).__init__()
        self.loss = nn.KLDivLoss(reduction='sum')

    def construct(self, stu_value_map, tea_value_map):
        return self.loss(
            f.softmax(stu_value_map, axis=1).log(),
            f.softmax(tea_value_map, axis=1)
        )
