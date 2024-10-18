import mindspore
from mindspore import nn
import mindspore.ops as ops
from mindspore import Tensor
class SMDMultiModel(nn.Cell):
    def __init__(self, tau=0.04, topk=1):
        super(SMDMultiModel, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.tau = tau
        self.topk = topk

    def construct(self, teacher_inputs, inputs, text_inputs, normalized=True):
        n = inputs.size(0)

        if normalized:
            inputs = ops.L2Normalize(inputs, axis=1)
            teacher_inputs = ops.L2Normalize(teacher_inputs, axis=1)
        # Teacher 的分布中，每一个样本对应的距离
        shape=(n,n)
        x1 = ops.pow(teacher_inputs, 2).sum(dim=1, keepdim=True)
        x1=Tensor(x1)
        x1=x1.broadcast_to(shape)
        x1=Tensor(x1)
        dist_t = x1 + x1.t()
        dist_t=Tensor(dist_t)
        dist_t=dist_t.addmm(teacher_inputs, ops.t(teacher_inputs), beta=1, alpha=-2)
        dist_t=Tensor(dist_t)
        dist_t = dist_t.clamp(min=1e-12)
        dist_t=ops.sqrt(dist_t)  # for numerical stability

        # Compute pairwise distance
        # Teacher 与 Student 的对应距离
        x1 = ops.pow(teacher_inputs, 2).sum(dim=1, keepdim=True)
        x1=Tensor(x1)
        x1=x1.broadcast_to(shape)
        x2 = ops.pow(inputs, 2).sum(dim=1, keepdim=True)
        x2=Tensor(x2)
        x2=x2.broadcast_to(shape)
        x2=Tensor(x2)
        dist = x1 + x2.t()
        dist=Tensor(dist)
        dist=dist.addmm(teacher_inputs, ops.t(inputs), beta=1, alpha=-2)
        dist = Tensor(dist)
        dist = dist.clamp(min=1e-12)
        dist=ops.sqrt(dist)  # for numerical stability

        # Compute image text distance
        x1 = ops.pow(inputs, 2).sum(dim=1, keepdim=True)
        x1=Tensor(x1)
        x1=x1.broadcast_to(shape)
        x2 = ops.pow(text_inputs, 2).sum(dim=1, keepdim=True)
        x2=Tensor(x2)
        x2=x2.broadcast_to(shape)
        x2=Tensor(x2)
        dist_text = x1 + x2.t()
        dist_text=Tensor(dist_text)
        dist_text=dist_text.addmm(teacher_inputs, ops.t(inputs), beta=1, alpha=-2)
        
        dist_text = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        text_positive = dist_text.diag()

        # For each anchor, find the hardest positive and negative
        negative_index = (dist_t > torch.diag(dist).expand(n, n).t()).float()
        negative = dist * negative_index
        negative[negative_index == 0] = 1e5
        positive_index = 1 - negative_index
        positive = dist * positive_index

        dist_an = torch.min(negative, dim=1)
        dist_ap = torch.max(positive, dim=1)

        an_t = torch.gather(dist_t, 1, dist_an.indices.unsqueeze(1)).squeeze()
        ap_t = torch.gather(dist_t, 1, dist_ap.indices.unsqueeze(1)).squeeze()

        weight_an = torch.clamp_min(an_t.detach() - dist_an.values.detach(), min=0.)
        weight_ap = torch.clamp_min(dist_ap.values.detach() - ap_t.detach(), min=0.)

        weight_dist_an = weight_an * dist_an.values / self.tau
        weight_dist_ap = weight_ap * dist_ap.values / self.tau
        weight_dist_text_positive = 1 * text_positive / self.tau

        logits = torch.cat([weight_dist_an.unsqueeze(-1),
                            weight_dist_ap.unsqueeze(-1),
                            weight_dist_text_positive.unspueeze(-1)], dim=1)
        labels = torch.zeros(weight_dist_an.shape[0], dtype=torch.long).cuda()

        return self.loss(logits, labels)
