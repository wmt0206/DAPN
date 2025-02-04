import torch
import torch.nn as nn
import torch.nn.functional as F


# # 余弦距离
# def cosine_dist(x, y):
#     # x: N x D
#     # y: M x D
#     x_norm = F.normalize(x, p=2, dim=1)
#     y_norm = F.normalize(y, p=2, dim=1)
#     cos_sim = torch.mm(x_norm, y_norm.t())
#     return 1-cos_sim
#
# class ProtoNet(nn.Module):
#     def __init__(self):
#         super(ProtoNet, self).__init__()
#
#     def forward(self, proto, query):
#         dists = cosine_dist(query, proto)
#         scores = -dists
#         return scores




def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)


class ProtoNet(nn.Module):
    def __init__(self):
        super(ProtoNet, self).__init__()

    def forward(self, proto, query):
        dists = euclidean_dist(query, proto)
        scores = -dists
        return scores



if __name__=='__main__':
    
    model = ProtoNet()
    x = torch.rand(5,576)
    y = torch.rand(75,576)
    scores = model(x, y)
    print(scores)