import torch
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(2 * 2 * 128, 256)

    def compute_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2, 2)

        x = x.flatten(1)

        x = F.relu(self.fc1(x))

        return x


class CNN_DUQ(Model):
    def __init__(
        self,
        input_size,
        num_classes, # 10
        embedding_size, # 256
        learnable_length_scale,
        length_scale,
        gamma,
    ):
        super().__init__()

        self.gamma = gamma
        self.W = nn.Parameter(
            torch.normal(torch.zeros(embedding_size, num_classes, 256), 0.05)
        ) # size(256, 10, 256)
        # Nc是在一个minibatch 中分配给类c的数据点的数量
        self.register_buffer("N", torch.ones(num_classes) * 12)
        # m/N是类中心
        self.register_buffer(
            "m", torch.normal(torch.zeros(embedding_size, num_classes), 1)
        )
        self.m = self.m * self.N.unsqueeze(0) # 增加维度

        # σ是超参数，有时称为⻓度尺度。
        if learnable_length_scale:
            self.sigma = nn.Parameter(torch.zeros(num_classes) + length_scale)
        else:
            self.sigma = length_scale

    # 更新类中心, 使⽤属于该类的数据点的特征向量的指数移动平均来更新
    def update_embeddings(self, x, y): # y是标签的onehot编码
        z = self.last_layer(self.compute_features(x)) # size(128,256,10)

        # normalizing value per class, assumes y is one_hot encoded
        self.N = self.gamma * self.N + (1 - self.gamma) * y.sum(0) 

        # compute sum of embeddings on class by class basis 特征和
        features_sum = torch.einsum("ijk,ik->jk", z, y) # i是样本数量，矩阵变换，Cjk = sigma_i(Zjk*Yk)

        self.m = self.gamma * self.m + (1 - self.gamma) * features_sum

    # 输出f*W
    def last_layer(self, z):
        # 每个类有一个可学习的权重矩阵W，i是样本数共128个样本，Z size(128,256); W size(256, 10, 256)
        z = torch.einsum("ij,mnj->imn", z, self.W) # 核心矩阵变换，Cmn = sigma_j(Zj*Wmnj) 输出 256*10
        return z # size(128,256,10)

    # 输出核
    def output_layer(self, z): 
        embeddings = self.m / self.N.unsqueeze(0)

        diff = z - embeddings.unsqueeze(0)
        distances = (-(diff ** 2)).mean(1).div(2 * self.sigma ** 2).exp()

        return distances

    def forward(self, x):
        z = self.last_layer(self.compute_features(x))
        y_pred = self.output_layer(z)

        return z, y_pred

