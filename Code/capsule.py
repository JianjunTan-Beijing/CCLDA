import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn import preprocessing
# from AE import MyDataset
import numpy as np


# # 定义编码器
# class Encoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim, latent_dim):
#         super(Encoder, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, latent_dim)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
#
# # 定义解码器
# class Decoder(nn.Module):
#     def __init__(self, latent_dim, hidden_dim, output_dim):
#         super(Decoder, self).__init__()
#         self.fc1 = nn.Linear(latent_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
#
# # 定义自编码器
# class Autoencoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
#         super(Autoencoder, self).__init__()
#         self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
#         self.decoder = Decoder(latent_dim, hidden_dim, output_dim)
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x


# CBAM注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        max_pool_out = self.max_pool(x).view([b, c])
        avg_pool_out = self.avg_pool(x).view([b, c])

        max_fc_out = self.fc(max_pool_out)
        avg_fc_out = self.fc(avg_pool_out)

        out = max_fc_out + avg_fc_out
        out = self.sigmoid(out).view([b, c, 1, 1])

        return out * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        y = torch.cat([max_pool, avg_pool], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)

# 胶囊网络

# 卷积层(1*1*1147*1147)——>(1*256*570*570)
class CapsuleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CapsuleConvLayer, self).__init__()
        self.pad = nn.ZeroPad2d(padding=(1, 1, 1, 1))
        # self.pad = nn.ReflectionPad2d(padding=(1, 1, 1, 1))
        # self.pad = nn.ReplicationPad2d(padding=(1, 1, 1, 1))
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=0,
                              bias=True)
        self.bn_conv = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.bn_conv(self.conv(self.pad(x))))
        # x = self.relu(self.conv(self.pad(x)))
        x = self.dropout(x)
        return x

# 初始胶囊层
def squash(x, dim=-1):    # 定义squash激活函数
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * x / (squared_norm.sqrt() + 1e-8)

# num_conv_units=8，(1*256*570*570)-->(1*256*281*281)-->(1*32*281*281)*8
class PrimaryCaps(nn.Module):
    def __init__(self, num_conv_units, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCaps, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels * num_conv_units,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=1)
        self.out_channels = out_channels

    def forward(self, x):
        # Shape of x: (batch_size, in_channels, height, weight)
        # Shape of out: num_capsules * (batch_size, out_channels, height, weight)
        out = self.conv(x)
        # Flatten out: (batch_size, num_capsules * height * weight, out_channels)
        # (1, 281*281*32=2526752, 8)
        batch_size = out.shape[0]
        return squash(out.contiguous().view(batch_size, -1, self.out_channels), dim=-1)

# 路由胶囊层
class DigitCaps(nn.Module):
    def __init__(self, in_dim, in_caps, num_caps, dim_caps, num_routing):
        """
        Initialize the layer.
        初始化图层。
        Args:
            in_dim: 		Dimensionality (i.e. length) of each capsule vector.
            每个胶囊向量的维数（即长度）。8
            in_caps: 		Number of input capsules if digits layer.
            如果数字层，输入胶囊的数量。32*2*32
            num_caps: 		Number of capsules in the capsule layer.
            胶囊层中的胶囊数量。2
            dim_caps: 		Dimensionality, i.e. length, of the output capsule vector.
            输出胶囊向量的维数，即长度。16
            num_routing:	Number of iterations during routing algorithm.
            路由算法期间的迭代次数
        """
        super(DigitCaps, self).__init__()
        self.in_dim = in_dim
        self.in_caps = in_caps
        self.num_caps = num_caps
        self.dim_caps = dim_caps
        self.num_routing = num_routing
        self.device = device
        self.W = nn.Parameter(0.01 * torch.randn(1, num_caps, in_caps, dim_caps, in_dim),
                              requires_grad=True)

    def forward(self, x):
        batch_size = x.size(0)
        # (batch_size, in_caps, in_dim) -> (batch_size, 1, in_caps, in_dim, 1)
        # (1, 2, 8)->(1, 1, 2, 8, 1)
        x = x.unsqueeze(1).unsqueeze(4)
        # W @ x =
        # (1, 2, 2526752, 16, 8) @ (1, 1, 2526752, 8, 1) = (1, 2, 2526752, 16, 1)
        # (1, num_caps, in_caps, dim_caps, in_dim) @ (batch_size, 1, in_caps, in_dim, 1) =
        # (batch_size, num_caps, in_caps, dim_caps, 1)
        u_hat = torch.matmul(self.W, x)
        # (batch_size, num_caps, in_caps, dim_caps)(1, 2, 2526752, 16)
        u_hat = u_hat.squeeze(-1)
        # detach u_hat during routing iterations to prevent gradients from flowing在路由迭代期间分离u_hat以防止梯度流动
        temp_u_hat = u_hat.detach()

        b = torch.zeros(batch_size, self.num_caps, self.in_caps, 1).to(self.device)

        for route_iter in range(self.num_routing - 1):
            # (batch_size, num_caps, in_caps, 1)(1, 2, 2526752, 1) -> Softmax along num_caps
            c = b.softmax(dim=1)

            # element-wise multiplication
            # (1, 2, 2526752, 1) * (1, 2526752, 1, 16)->(1, 2, 2526752, 16)->(1, 2, 16)
            # (batch_size, num_caps, in_caps, 1) * (batch_size, in_caps, num_caps, dim_caps) ->
            # (batch_size, num_caps, in_caps, dim_caps) sum across in_caps ->
            # (batch_size, num_caps, dim_caps)
            s = (c * temp_u_hat).sum(dim=2)
            # apply "squashing" non-linearity along dim_caps
            v = squash(s)
            # dot product agreement between the current output vj and the prediction uj|i
            # (1, 2, 2526752, 16) @ (1, 2, 16, 1)-> (1, 2, 2526752, 1)
            # (batch_size, num_caps, in_caps, dim_caps) @ (batch_size, num_caps, dim_caps, 1)
            # -> (batch_size, num_caps, in_caps, 1)
            uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))
            b += uv

        # last iteration is done on the original u_hat, without the routing weights update
        c = b.softmax(dim=1)
        s = (c * u_hat).sum(dim=2)
        # apply "squashing" non-linearity along dim_caps
        v = squash(s)

        return v


device =torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CapsNet(nn.Module):
    """Basic implementation of capsule network layer."""

    def __init__(self):
        super(CapsNet, self).__init__()

        # Conv2d layer
        self.capsuleconv_layer = CapsuleConvLayer(1, 256)

        # Attention layer
        self.channel_attention = ChannelAttention(256, reduction_ratio=64)
        self.spatial_attention = SpatialAttention()

        # Primary capsule
        self.primary_caps = PrimaryCaps(num_conv_units=32,
                                        in_channels=256,
                                        out_channels=8,
                                        kernel_size=3,
                                        stride=1)

        # Digit capsule
        self.digit_caps = DigitCaps(in_dim=8,   # out_channels
                                    in_caps=32 * 2 * 32,
                                    num_caps=2,
                                    dim_caps=16,
                                    num_routing=3)

        # Reconstruction layer
        self.decoder = nn.Sequential(
            nn.Linear(16 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 2*32),
            nn.Sigmoid())

    def forward(self, x):
        out = self.capsuleconv_layer(x)
        # out = self.channel_attention(out)
        # out = self.spatial_attention(out)
        out = self.primary_caps(out)
        out = self.digit_caps(out)

        # Shape of logits: (batch_size, num_capsules)
        logits = torch.norm(out, dim=-1)
        threshold = 0.5
        pred = (logits > threshold).float()

        # Reconstruction
        batch_size = out.shape[0]
        reconstruction = self.decoder((out * pred.unsqueeze(2)).contiguous().view(batch_size, -1))

        return logits, reconstruction

# 损失函数
class CapsuleLoss(nn.Module):
    """Combine margin loss & reconstruction loss of capsule network."""
    """胶囊网络综合边缘损失和重构损失"""

    def __init__(self, upper_bound=0.9, lower_bound=0.1, lmda=0.5):
        super(CapsuleLoss, self).__init__()
        self.upper = upper_bound
        self.lower = lower_bound
        self.lmda = lmda
        self.reconstruction_loss_scalar = 0.0001
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, images, labels, logits, reconstructions):
        # Shape of left / right / labels: (batch_size, num_classes)
        left = (self.upper - logits).relu() ** 2  # True negative
        right = (logits - self.lower).relu() ** 2  # False positive
        margin_loss = torch.sum(labels * left[:, 1]) + self.lmda * torch.sum((1 - labels) * right[:, 1])
        # margin_loss = labels * left + (1 - labels) * right

        # Reconstruction loss
        reconstruction_loss = self.mse(reconstructions.contiguous().view(images.shape), images)

        # Combine two losses
        return margin_loss + self.reconstruction_loss_scalar * reconstruction_loss

