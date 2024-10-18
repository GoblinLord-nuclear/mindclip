from mindspore import nn
from mindspore import ops
class HardLabel(nn.Cell):
    def __init__(self):
        super(HardLabel, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='mean')
    def construct(self, stu_logits):
        label = ops.arange(stu_logits.shape[0], device=stu_logits.device)
        return self.loss(stu_logits, label)