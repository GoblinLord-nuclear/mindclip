from mindspore import nn
class EmbedMSELoss(nn.Cell):
    def __init__(self):
        super(EmbedMSELoss, self).__init__()
        self.loss = nn.MSELoss()

    def construct(self, stu_embedding, tea_embedding):
        return self.loss(stu_embedding, tea_embedding)