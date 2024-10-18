import mindspore.numpy as mnp
from mindspore import nn
from mindspore import Tensor


class OutCosLoss(nn.Cell):
    def __init__(self):
        super().__init__()
        self.loss = nn.CosineEmbeddingLoss()

    def construct(self, stu_out, tea_out):

        y = mnp.ones(stu_out.shape[0])
        
      
        return self.loss(stu_out, tea_out, y)
