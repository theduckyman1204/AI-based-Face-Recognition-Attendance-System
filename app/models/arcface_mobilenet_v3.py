import math
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large


def load_mobilenet_v3_backbone(pretrained_weight=None):
    model = mobilenet_v3_large(weights=None)
    if pretrained_weight is not None:
        model.load_state_dict(pretrained_weight)

    features = model.features
    features.append(model.avgpool)
    return features


class ArcMarginProduct(nn.Module):
    def __init__(self, feature_dim, num_classes, s=64.0, m=0.5):
        super().__init__()
        self.in_features = feature_dim
        self.out_features = num_classes
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(self.out_features, self.in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, inputs, label):
        cosine = nn.functional.linear(inputs, nn.functional.normalize(self.weight))
        sine = torch.sqrt(1.0 - cosine ** 2 + 1e-7)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class ArcFace(nn.Module):
    def __init__(self, num_classes, feature_dim=256, backbone_weight=None):
        super(ArcFace, self).__init__()
        self.backbone = load_mobilenet_v3_backbone(backbone_weight)
        self.embedding = nn.Linear(960, feature_dim)
        self.arcface = ArcMarginProduct(feature_dim, num_classes)

    def forward(self, x, label=None):
        out = self.backbone(x)
        out = out.view(-1, 960)
        out = nn.functional.normalize(self.embedding(out))

        if self.training and label is not None:
            logits = self.arcface(out, label)
            return logits
        else:
            # dùng embedding cho nhận diện
            return out
