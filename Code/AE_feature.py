import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn import preprocessing
import numpy as np
import os
import random

os.chdir(os.path.dirname(__file__))

'模型结构'
class Encoder(torch.nn.Module):
    # 编码器，将input_size维度数据压缩为latent_size维度
    def __init__(self, input_size, hidden_size, latent_size):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, latent_size)

    def forward(self, x):    # x: bs,input_size
        x = F.relu(self.linear1(x))     # -> bs,hidden_size
        x = self.linear2(x)     # -> bs,latent_size
        return x

class Decoder(torch.nn.Module):
    # 解码器，将latent_size维度的压缩数据转换为output_size维度的数据
    def __init__(self, latent_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(latent_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):     # x:bs,latent_size
        x = F.relu(self.linear1(x))     # ->bs,hidden_size
        x = torch.sigmoid(self.linear2(x))     # ->bs,output_size
        return x

class AE(torch.nn.Module):
    # 将编码器解码器组合，数据先后通过编码器、解码器处理
    def __init__(self, input_size, output_size, latent_size, hidden_size):
        super(AE, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, latent_size)
        self.decoder = Decoder(latent_size, hidden_size, output_size)
    def forward(self, x):     # x: bs,input_size
        feat = self.encoder(x)     # feat: bs,latent_size
        re_x = self.decoder(feat)     # re_x: bs, output_size
        return re_x

# 损失函数
# 交叉熵，衡量各个像素原始数据与重构数据的误差
loss_BCE = torch.nn.BCELoss(reduction='sum')
# 均方误差可作为交叉熵替代使用.衡量各个像素原始数据与重构数据的误差
loss_MSE = torch.nn.MSELoss(reduction='sum')

'超参数及构造模型'
# 模型参数
latent_size = 2*32     # 压缩后的特征维度
hidden_size = 2*1024     # encoder和decoder中间层的维度
input_size = output_size = 2*1305    # 原始图片和生成图片的维度1147/1305

# 训练参数
epochs = 250    # 训练时期
batch_size = 64    # 每步训练样本数
learning_rate = 0.001    # 学习率
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')    # 训练设备

# 确定模型，导入已训练模型（如有）
# modelname = 'ae.pth'
model = AE(input_size, output_size, latent_size, hidden_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# try:
#     model.load_state_dict(torch.load(modelname))
#     print('[INFO] Load Model complete')
# except:
#     pass

# 数据准备
def get_data(lncSiNet, diSiNet, lnc_di, lnc_mi, mi_di, ij):
    data = []
    for i, j in ij:
        part1 = np.vstack([lncSiNet.iloc[i], lnc_di.iloc[:, j]])
        part2 = np.vstack([lnc_di.iloc[i], diSiNet.iloc[j]])
        part3 = np.vstack([lnc_mi.iloc[i], mi_di.iloc[:, j]])
        Pi_j = np.hstack([part1, part2, part3])
        data.append(Pi_j)
    return np.array(data)


positive_ij = np.load('dataset2/positive_ij.npy')    # Fu
negative_ij = np.load('dataset2/negative_ij.npy')
positive5foldsidx = np.load('dataset2/positive5foldsidx.npy', allow_pickle=True)
negative5foldsidx = np.load('dataset2/negative5foldsidx.npy', allow_pickle=True)
lncSiNet = np.load('dataset2/similarity_network/LS.npy')
lncSiNet = pd.DataFrame(lncSiNet)
diSiNet = np.load('dataset2/similarity_network/DS.npy')
diSiNet = pd.DataFrame(diSiNet)
lnc_di = pd.read_csv('dataset2/lnc_di.csv')
lnc_di = pd.DataFrame(lnc_di)
lnc_di = lnc_di.iloc[:, 1:]
lnc_mi = pd.read_csv('dataset2/lnc_mi.csv')
lnc_mi = pd.DataFrame(lnc_mi)
lnc_mi = lnc_mi.iloc[:, 1:]
mi_di = pd.read_csv('dataset2/mi_di.csv')
mi_di = pd.DataFrame(mi_di)
mi_di = mi_di.iloc[:, 1:]


class myDataset_train(Dataset):
    def __init__(self, fold) -> None:
        super().__init__()

        positive_train_ij = positive_ij[positive5foldsidx[fold]['train']]
        negative_train_ij = negative_ij[negative5foldsidx[fold]['train']]
        positive_train_data = torch.Tensor(
            get_data(lncSiNet, diSiNet, lnc_di, lnc_mi, mi_di, positive_train_ij))
        negative_train_data = torch.Tensor(
            get_data(lncSiNet, diSiNet, lnc_di, lnc_mi, mi_di, negative_train_ij))
        self.data = torch.cat((positive_train_data, negative_train_data))
        self.target = torch.Tensor([1] * len(positive_train_ij) + [0] * len(negative_train_ij))

        print('Finished reading the train set of Dataset ({} samples found, each dim = {})'
              .format(len(self.data), self.data.shape[1:]))

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)

class myDataset_test(Dataset):
    def __init__(self, fold) -> None:
        super().__init__()

        positive_test_ij = positive_ij[positive5foldsidx[fold]['test']]
        negative_test_ij = negative_ij[negative5foldsidx[fold]['test']]

        # Randomly select the same number of negative samples as positive samples
        num_negative_samples = len(positive_test_ij)
        selected_negative_indices = random.sample(range(len(negative_test_ij)), num_negative_samples)
        negative_test_ij = [negative_test_ij[i] for i in selected_negative_indices]

        positive_test_data = torch.Tensor(
            get_data(lncSiNet, diSiNet, lnc_di, lnc_mi, mi_di, positive_test_ij))
        negative_test_data = torch.Tensor(
            get_data(lncSiNet, diSiNet, lnc_di, lnc_mi, mi_di, negative_test_ij))
        self.data = torch.cat((positive_test_data, negative_test_data))
        self.target = torch.Tensor([1] * len(positive_test_ij) + [0] * len(negative_test_ij))

        print('Finished reading the test set of Dataset ({} samples found, each dim = {})'
              .format(len(self.data), self.data.shape[1:]))

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)

# mly = myDataset_train(0) + myDataset_test(0)
# mly = torch.utils.data.DataLoader(
#     dataset=mly,
#     batch_size=batch_size, shuffle=True)
# print(mly.shape)

'训练模型'
fold = 4
dataset_loader = torch.utils.data.DataLoader(
    dataset=myDataset_train(fold) + myDataset_test(fold),
    batch_size=batch_size, shuffle=True)
train_loader = torch.utils.data.DataLoader(
    dataset=myDataset_train(fold),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=myDataset_test(fold),
    batch_size=batch_size, shuffle=False)

# 训练及测试
loss_history = {'train': [], 'eval': []}
for epoch in range(epochs):
    # 训练
    model.train()
    # 每个epoch重置损失，设置进度条
    train_loss = 0
    train_nsample = 0
    t = tqdm(dataset_loader, desc=f'[train]epoch:{epoch}')
    for images, labels in t:     # imgs:(bs,28,28)
        bs = images.shape[0]
        # 获取数据
        images = images.to(device).view(bs, input_size)    # imgs:(bs,28*28)
        # 模型运算
        re_images = model(images)
        # 计算损失
        loss = loss_MSE(re_images, images)    # 重构与原始数据的差距(也可使用loss_MSE)
        # 反向传播、参数优化，重置
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # 计算平均损失，设置进度条
        train_loss += loss.item()
        train_nsample += bs
        t.set_postfix({'loss': train_loss/train_nsample})
    # 每个epoch记录总损失
    loss_history['train'].append(train_loss/train_nsample)
    # 显示每个epoch的loss变化
    plt.plot(range(epoch + 1), loss_history['train'])
    # plt.plot(range(epoch+1), loss_history['eval'])
plt.show()
# 保存训练好的模型
torch.save(model, 'dataset2/AE_train.pth')    #

# 测试
# 训练集降维
model.eval()
with torch.no_grad():
    train_data = []  # 用于存储特征和标签
    for images, labels in train_loader:
        bs = images.shape[0]
        images = images.to(device).view(bs, input_size)
        train_feature = model.encoder(images)
        train_feature = train_feature.view(bs, 2, -1)
        train_data.append((train_feature, labels))  # 添加特征和标签的元组
    train_features, train_labels = zip(*train_data)  # 拆分特征和标签
    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
train_features = train_features.cpu().numpy()
train_labels = train_labels.cpu().numpy()
print("Train Features Shape:", train_features.shape)
print("Train Labels Shape:", train_labels.shape)

# 保存 train_data 为 .npy 文件
np.save('dataset2/feature/train_data('+str(fold)+').npy', {'features': train_features, 'labels': train_labels})

# 测试集降维
with torch.no_grad():
    test_data = []  # 用于存储特征和标签
    for images, labels in test_loader:
        bs = images.shape[0]
        images = images.to(device).view(bs, input_size)
        test_feature = model.encoder(images)
        test_feature = test_feature.view(bs, 2, -1)
        test_data.append((test_feature, labels))  # 添加特征和标签的元组
    test_features, test_labels = zip(*test_data)  # 拆分特征和标签
    test_features = torch.cat(test_features, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
test_features = test_features.cpu().numpy()
test_labels = test_labels.cpu().numpy()
print("Test Features Shape:", test_features.shape)
print("Test Labels Shape:", test_labels.shape)

# 保存 test_data 为 .npy 文件
np.save('dataset2/feature/test_data('+str(fold)+').npy', {'features': test_features, 'labels': test_labels})


#     model.eval()
#     # 每个epoch重置损失，设置进度条
#     test_loss = 0
#     test_nsample = 0
#     e = tqdm(test_loader, desc=f'[eval]epoch:{epoch}')
#     for images, labels in e:
#         bs = images.shape[0]
#         # 获取数据
#         images = images.to(device).view(bs, input_size)
#         # 模型运算
#         re_images = model(images)
#         # 计算损失
#         loss = loss_MSE(re_images, images)
#         # 计算平均损失，设置进度条
#         test_loss += loss.item()
#         test_nsample += bs
#         e.set_postfix({'loss': test_loss/test_nsample})
#     # 每个epoch记录总损失
#     loss_history['eval'].append(test_loss/test_nsample)
#
#     # #展示效果
#     # #将测试步骤中的数据、重构数据绘图
#     # concat = torch.cat((imgs[0].view(28, 28),
#     #         re_imgs[0].view( 28, 28)), 1)
#     # plt.matshow(concat.cpu().detach().numpy())
#     # plt.show()
#
    # # 显示每个epoch的loss变化
    # plt.plot(range(epoch+1), loss_history['train'])
    # # plt.plot(range(epoch+1), loss_history['eval'])
    # plt.show()
    # # 存储模型
    # # torch.save(model.state_dict(), modelname)
#
# '调用模型'
# # 对数据集
# dataset_train = myDataset_train(fold)
# dataset_test = myDataset_test(fold)
# # 取一组数据
# # raw = dataset_test[0].view(1, -1)    # raw: bs,28,28->bs,28*28
# # 用encoder压缩数据
# # raw = raw.to(device)
# # feat = model.encoder(raw)
# # 展示数据及维度
# # print(raw.shape, '->', feat.shape)
#
# features_train = []
# # 遍历数据集并调整每个样本的形状
# for i in range(len(dataset_train)):
#     sample = dataset_train[i][0]  # 获取样本的输入数据
#     feature_train = sample.view(1, -1)  # 调整形状为 (1, -1)
#     features_train.append(feature_train)
# # 使用 torch.cat 将所有样本连接成一个张量
# all_features_train = torch.cat(features_train, dim=0)
# # 打印结果的形状
# print(all_features_train.shape)
#
# all_features_train = all_features_train.to(device)
# train = model.encoder(all_features_train)
# print(all_features_train.shape, '->', train.shape)


