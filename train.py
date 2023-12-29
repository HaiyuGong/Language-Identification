import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import time

from model import SimpleCNN
from datasetLoader import MelSpectrogramDataset

model = SimpleCNN((400,80),2)
model=model.cuda()
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 假设你有一个 MelSpectrogramDataset 的实例 dataset
dataset = MelSpectrogramDataset('data/train_data')

# 获取所有样本的索引
all_indices = list(range(len(dataset)))

# 随机划分训练集和验证集
train_indices, val_indices = train_test_split(all_indices, test_size=0.2, random_state=42)

# 定义训练集和验证集的 DataLoader
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True,num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=100, shuffle=False,num_workers=4)

# 训练模型
num_epochs = 50
losses=np.zeros((num_epochs,3))
best_acc_val = 0
best_acc_train = 0
start_time=time.time()
for epoch in range(num_epochs):
    loss_list=[]
    accuracy_list_train=[]
    accuracy_list_val=[]

    for mel_spectrogram, label,_ in train_dataloader:
        mel_spectrogram = mel_spectrogram.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        outputs = model(mel_spectrogram)
        # outputs=torch.softmax(outputs,dim=1)
        loss = criterion(outputs, label)
        _, predicted = torch.max(outputs, 1)
        accuracy_list_train+=(predicted == label).tolist()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
    for mel_spectrogram, label,_ in val_dataloader:
        mel_spectrogram = mel_spectrogram.cuda()
        label = label.cuda()
        outputs = model(mel_spectrogram)
        # outputs=torch.softmax(outputs,dim=1)
        _, predicted = torch.max(outputs, 1)
        accuracy_list_val+=(predicted == label).tolist()
    loss_avg=np.mean(np.array(loss_list))
    acc_val=np.mean(np.array(accuracy_list_val))
    acc_train=np.mean(np.array(accuracy_list_train))
    if acc_val>best_acc_val:
        best_acc_val=acc_val
        best_acc_train=acc_train
        torch.save(model.state_dict(), './saved_models/model_best1.pth')

    losses[epoch,:]=np.array([loss_avg,acc_train,acc_val])
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss_avg:.4f}, Accuracy_train: {acc_train:.4f}, Accuracy_val: {acc_val:.4f}")
print("-"*20)
print('time:',time.time()-start_time)
np.save('losses.npy',losses)
print('best_acc_train:',best_acc_train,'best_acc_val:',best_acc_val)
