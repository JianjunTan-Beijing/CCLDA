import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from capsule import CapsNet, CapsuleLoss
import pandas as pd
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, \
    precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef, \
    roc_curve, precision_recall_curve, auc as sklearn_auc
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time

# Check cuda availability
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
    # Load model
    model = CapsNet().to(device)
    criterion = CapsuleLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00008)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

    class CustomDataset(Dataset):
        def __init__(self, data_file):
            data = np.load(data_file, allow_pickle=True)
            self.features = data.item()['features']
            self.labels = data.item()['labels']

        def __len__(self):
            return len(self.features)

        def __getitem__(self, index):
            features = self.features[index]  # 获取原始特征
            features = torch.tensor(features).view(1, 2, 32)  # 转换为（1，2，32）张量
            labels = torch.tensor(self.labels[index])
            # labels = int(labels)
            return features, labels

    # 文件路径
    fold = 2
    train_data_file = 'dataset2/feature/train_data('+str(fold)+').npy'
    test_data_file = 'dataset2/feature/test_data('+str(fold)+').npy'

    # 创建训练数据集和测试数据集
    train_dataset = CustomDataset(train_data_file)
    test_dataset = CustomDataset(test_data_file)

    BATCH_SIZE = 128
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False)

    # Train
    # EPOCHES = 20
    # model.train()
    # for ep in range(EPOCHES):
    #     batch_id = 1
    #     correct, total, total_loss = 0, 0, 0.
    #     for images, labels in train_loader:
    #         optimizer.zero_grad()
    #         images = images.to(device)
    #         labels = labels.to(device)
    #         logits, reconstruction = model(images)
    #
    #         # Compute loss & accuracy
    #         loss = criterion(images, labels, logits, reconstruction)
    #         correct += torch.sum(
    #             torch.argmax(logits, dim=1) == labels).item()
    #         total += len(labels)
    #         accuracy = correct / total
    #         total_loss += loss
    #         loss.backward()
    #         optimizer.step()
    #         print('Epoch {}, batch {}, loss: {}, accuracy: {}'.format(ep + 1,
    #                                                                   batch_id,
    #                                                                   total_loss / batch_id,
    #                                                                   accuracy))
    #         batch_id += 1
    #     scheduler.step(ep)
    #     print('Total loss for epoch {}: {}'.format(ep + 1, total_loss))
    EPOCHES = 180
    model.train()
    for ep in range(EPOCHES):
        batch_id = 1
        correct, total, total_loss = 0, 0, 0.

        # 使用tqdm包装train_loader以显示进度条
        t = tqdm(train_loader, desc=f'Epoch {ep + 1}')
        for images, labels in t:
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            logits, reconstruction = model(images)

            # 计算损失和准确性
            loss = criterion(images, labels, logits, reconstruction)
            correct += torch.sum(torch.argmax(logits, dim=1) == labels).item()
            total += len(labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            batch_id += 1
            t.set_postfix({'loss': total_loss/batch_id}, accuracy=correct / total)
        scheduler.step(ep)
    torch.save(model, 'dataset2/Capsule_train.pth')

    # Eval
    model.eval()
    correct, total = 0, 0
    true_labels = []
    predicted_scores = []
    for images, labels in test_loader:
        # Add channels = 1
        images = images.to(device)
        labels = labels.to(device)
        # Categogrical encoding
        logits, reconstructions = model(images)
        predicted_scores.append(logits[:, 1].detach().cpu().numpy())  # Assuming the second class is positive
        true_labels.append(labels.detach().cpu().numpy())
        pred_labels = torch.argmax(logits, dim=1)
        correct += torch.sum(pred_labels == labels).item()
        total += len(labels)
    print('Accuracy: {}'.format(correct / total))

    # Concatenate lists into NumPy arrays
    true_labels = np.concatenate(true_labels)
    predicted_scores = np.concatenate(predicted_scores)

    # folder_path = 'dataset2/result/' + str(fold) + 'fold'
    folder_path = 'dataset2/result ablation/CCLDA-c'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    np.save(os.path.join(folder_path, 'true_labels.npy'), true_labels)
    np.save(os.path.join(folder_path, 'predicted_scores.npy'), predicted_scores)

    result_df = pd.DataFrame({'True Labels': true_labels, 'Predicted Scores': predicted_scores})
    excel_filename = os.path.join(folder_path, 'predicted_scores.xlsx')
    result_df.to_excel(excel_filename, index=False)

    # Calculate AUC and AUPR
    auc = roc_auc_score(true_labels, predicted_scores)
    aupr = average_precision_score(true_labels, predicted_scores)

    # Calculate F1 score
    threshold = 0.5  # You can adjust the threshold as needed
    predicted_labels = (predicted_scores >= threshold).astype(int)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    kappa = cohen_kappa_score(true_labels, predicted_labels)
    mcc = matthews_corrcoef(true_labels, predicted_labels)

    # Print the results
    print('AUC: {:.4f}'.format(auc))
    print('AUPR: {:.4f}'.format(aupr))
    print('Precision: {:.4f}'.format(precision))
    print('Recall: {:.4f}'.format(recall))
    print('F1 Score: {:.4f}'.format(f1))
    print('Kappa: {:.4f}'.format(kappa))
    print('MCC: {:.4f}'.format(mcc))

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(true_labels, predicted_scores)
    roc_auc = sklearn_auc(fpr, tpr)

    # Calculate PR curve
    precision, recall, _ = precision_recall_curve(true_labels, predicted_scores)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    roc_file_path = os.path.join(folder_path, "roc_curve("+str(fold)+").png")
    plt.savefig(roc_file_path)

    # Plot PR curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUPR = {aupr:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) Curve')
    plt.legend(loc='lower left')
    pr_file_path = os.path.join(folder_path, "pr_curve("+str(fold)+").png")
    plt.savefig(pr_file_path)

    plt.show()

    # Save model
    # torch.save(model.state_dict(), './model/capsnet_ep{}_acc{}.pt'.format(EPOCHES, correct / total))

start = time.perf_counter()

if __name__ == '__main__':
    main()

end = time.perf_counter()
runTime = (end - start)/60
print("运行时间：", runTime, "分")
