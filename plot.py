import matplotlib.pyplot as plt
import numpy as np
import os


# plot MelSpectrogram.png
mel_spectrogram = np.load('data/train_data/language_0/M003_075.npy')
print(mel_spectrogram.shape)
min_value = np.min(mel_spectrogram)
if mel_spectrogram.shape[0] < 400:
    # 填充到最大长度
    pad_length = 400 - mel_spectrogram.shape[0]
    mel_data = np.pad(mel_spectrogram, ((0, pad_length), (0, 0)), mode='constant', constant_values=min_value)
elif mel_spectrogram.shape[0] > 400:
    # 截断到最大长度
    mel_data = mel_spectrogram[:400, :]

# 可视化梅尔频谱
plt.figure(figsize=(6, 6))
plt.subplot(2, 1, 1)
plt.imshow(mel_spectrogram.T, interpolation='nearest')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Mel Spectrogram')
# 设置 x 和 y 轴的同比例
# plt.axis('equal')
plt.subplot(2, 1, 2)
plt.imshow(mel_data.T, interpolation='nearest')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Mel Spectrogram')
plt.savefig('img/MelSpectrogram.png')
plt.show()



# plot the histogram of the data
path1 = 'data/train_data/language_0'
path2 = 'data/train_data/language_1'
path3 = 'data/test_data'
file_names1 = os.listdir(path1)
file_names2 = os.listdir(path2)
file_names3 = os.listdir(path3)
len_list1=[]
len_list2=[]
len_list3=[]
for file_name in file_names1:
    mel_spectrogram = np.load(os.path.join(path1,file_name))
    len_list1.append(mel_spectrogram.shape[0])
for file_name in file_names2:
    mel_spectrogram = np.load(os.path.join(path2,file_name))
    len_list2.append(mel_spectrogram.shape[0])
for file_name in file_names3:
    mel_spectrogram = np.load(os.path.join(path3,file_name))
    len_list3.append(mel_spectrogram.shape[0])
plt.figure(figsize=(15, 5))
plt.subplot(1,3,1)
plt.hist(len_list1, bins=30, color='steelblue' )
plt.title('language_0')
plt.subplot(1,3,2)
plt.hist(len_list2, bins=30, color='steelblue' )
plt.title('language_1')
plt.subplot(1,3,3)
plt.hist(len_list3, bins=30, color='steelblue' )
plt.title('test_data')
plt.savefig('img/hist.png')
plt.show()

# plot train_prococess.png
losses=np.load('losses.npy')
epochs = range(1, losses.shape[0]+1)
fig, ax = plt.subplots()
ax.plot(epochs, losses[:,0], label='Loss')
ax.plot(epochs, losses[:,1], label='Train Acc')
ax.plot(epochs, losses[:,2], label='Val Acc')
ax.set_xlabel('Epochs')
ax.set_ylabel('Metrics')
ax.set_title('Training Loss & Accuracy')
ax.legend()
plt.savefig('img/train_prococess.png')
plt.show()