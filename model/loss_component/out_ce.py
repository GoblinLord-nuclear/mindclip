import mindspore
from mindspore import nn
from mindspore.ops import operations as P

class OutCELoss(nn.Cell):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction='mean')
        self.softmax = P.Softmax(axis=1)
        
    def construct(self, stu_out, tea_out):
         return self.loss(
            stu_out,  # [batch, out_dim]
            self.softmax(tea_out)
        )

