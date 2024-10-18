import mindspore
from mindspore import nn
import mindspore.ops as ops
from mindspore import Tensor
class SMD(nn.Cell):

    def __init__(self, tau=0.04):
        super(SMD, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.tau = tau

    def construct(self, teacher_inputs, inputs, normalized=True):
        n = inputs[0]

        if normalized:
            inputs = ops.L2Normalize(inputs, axis=1)
            teacher_inputs = ops.L2Normalize(teacher_inputs, axis=1)
        # 计算teacher inputs的每一个样本之间的距离 dist_t.shape [n, n]
        shape=(n,n)
        x1 = ops.pow(teacher_inputs, 2).sum(dim=1, keepdim=True)
        x1=Tensor(x1)
        x1=x1.broadcast_to(shape)
        dist_t = x1 + ops.t(x1)
        teacher_inputs=Tensor(teacher_inputs)
        dist_t=Tensor(dist_t)
        dist_t = dist_t.addmm(teacher_inputs, ops.t(teacher_inputs), beta=1, alpha=-2)
        dist_t=Tensor(dist_t)
        dist_t = dist_t.clamp(min=1e-12)
        dist_t=ops.sqrt(dist_t)  # for numerical stability

        # Compute pairwise distance
        # 计算inputs之间的点和teacher inputs每一个点的距离 dist.shape [n, n]
        x1 = ops.pow(teacher_inputs, 2).sum(dim=1, keepdim=True)
        x1=Tensor(x1)
        x1=x1.broadcast_to(shape)
        x2 = ops.pow(inputs, 2).sum(dim=1, keepdim=True)
        x2=Tensor(x2)
        x2=x2.broadcast_to(shape)
        x2=Tensor(x2)
        dist = x1 + x2.t()
        dist=Tensor(dist)
        dist=dist.addmm(teacher_inputs, inputs.t(), beta=1, alpha=-2)
        dist=Tensor(dist)
        dist = dist.clamp(min=1e-12)
        dist=ops.sqrt(dist)  # for numerical stability

        # For each anchor, find the hardest positive and negative
        t_dist=ops.diag(dist_t).broadcast_to(shape)
        t_dist=Tensor(t_dist)
        t_dist=t_dist.t()
        negative_index = (dist_t > t_dist)
        negative_index=Tensor(negative_index)
        negative_index=negative_index.astype(mindspore.float32)
        negative = dist * negative_index
        negative[negative_index == 0] = 1e5
        positive_index = 1 - negative_index
        positive = dist * positive_index

        dist_an = ops.min(negative, axis=1)
        dist_ap = ops.max(positive, axisw=1)

        an_t = ops.gather(dist_t, 1, dist_an.indices.unsqueeze(1)).squeeze()
        ap_t = torch.gather(dist_t, 1, dist_ap.indices.unsqueeze(1)).squeeze()

        weight_an = torch.clamp_min(an_t.detach() - dist_an.values.detach(), min=0.)
        weight_ap = torch.clamp_min(dist_ap.values.detach() - ap_t.detach(), min=0.)

        weight_dist_an = weight_an * dist_an.values / self.tau
        weight_dist_ap = weight_ap * dist_ap.values / self.tau

        logits = torch.cat([weight_dist_an.unsqueeze(-1), weight_dist_ap.unsqueeze(-1)], dim=1)
        labels = torch.zeros(weight_dist_an.shape[0], dtype=torch.long).cuda()

        return self.loss(logits, labels)
