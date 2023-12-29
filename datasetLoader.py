import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

class MelSpectrogramDataset(Dataset):
    def __init__(self, dir_path,mode='train',max_len=400):
        self.filename_list=[]
        self.mode=mode  
        self.max_len=max_len
        if self.mode == 'train':
            for language in os.listdir(dir_path):
                for filename in os.listdir(os.path.join(dir_path, language)):
                    self.filename_list.append(tuple((os.path.join(dir_path, language, filename),int(language[-1]))))
        elif self.mode == 'test':
            for filename in os.listdir(dir_path):
                self.filename_list.append(tuple((os.path.join(dir_path, filename),)))

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, idx):
        file_path=self.filename_list[idx][0]
        mel_spectrogram = torch.tensor(np.load(file_path), dtype=torch.float32)
        max_value=torch.max(mel_spectrogram)
        min_value=torch.min(mel_spectrogram)
        mel_spectrogram=(mel_spectrogram-min_value)/(max_value-min_value)
        if mel_spectrogram.size(0) > self.max_len:
            mel_spectrogram = mel_spectrogram[:self.max_len, :]
        else:
            mel_spectrogram = torch.cat((mel_spectrogram, torch.zeros(self.max_len - mel_spectrogram.size(0), mel_spectrogram.size(1))), dim=0)
        if self.mode == 'train':
            # label = torch.tensor([0.0,1.0]) if self.filename_list[idx][1]==1 else torch.tensor([1.0,0.0])
            label = torch.tensor(self.filename_list[idx][1])
            return mel_spectrogram,label,os.path.basename(file_path)
        elif self.mode == 'test':
            return mel_spectrogram,os.path.basename(file_path)

if __name__ == '__main__':
    train_dataset = MelSpectrogramDataset('./data/train_data/', mode='train')
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    for i, (mel_spectrogram, label, filename) in enumerate(train_loader):
        print("mel_spectrogram's shape",mel_spectrogram.shape)
        print("label:",label)
        print("filename:",filename)
        break
    test_dataset = MelSpectrogramDataset('./data/test_data/', mode='test')
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    for i, (mel_spectrogram, filename) in enumerate(test_loader):
        print("mel_spectrogram's shape",mel_spectrogram.shape)
        print("filename:",filename)
        break
    