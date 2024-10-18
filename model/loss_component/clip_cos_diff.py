from mindspore import nn
from mindspore.ops import operations as P

def get_neg_element(x):
    m, n = P.Shape()(x)
    assert m == n
    return P.Reshape()(x, (n - 1, n + 1))[:, 1:].flatten() # 牛的，刚好除掉对角线

class CLIPCosDiff(nn.Cell):
    def __init__(self):
        super(CLIPCosDiff, self).__init__()
        self.relu = nn.ReLU()
        self.mean = P.ReduceMean()
        self.diag = P.DiagPart()
    def forward(self, stu_logits, tea_logits):
        stu_pos_dis = self.diag(stu_logits)
        tea_pos_dis = self.diag(tea_logits)
        pos_loss = self.mean(self.relu(tea_pos_dis - stu_pos_dis))
        stu_neg_dis = get_neg_element(stu_logits)
        tea_neg_dis = get_neg_element(tea_logits)
        neg_loss = self.mean(self.relu(stu_neg_dis - tea_neg_dis))
        return neg_loss + pos_loss
