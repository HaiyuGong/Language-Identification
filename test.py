import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import SimpleCNN
from datasetLoader import MelSpectrogramDataset

df=pd.read_csv('data/test.csv')
df.head()

test_dataset = MelSpectrogramDataset('data/test_data',mode='test')
test_dataloader = DataLoader(test_dataset, batch_size=90, shuffle=False)

input_size = (400, 80)
num_classes = 2
model = SimpleCNN(input_size, num_classes)
model = model.cuda()
model.load_state_dict(torch.load('./saved_models/model_best.pth'))
with torch.no_grad():
    model.eval()
    for mel_spectrogram,filename in test_dataloader:
        mel_spectrogram = mel_spectrogram.cuda()
        outputs = model(mel_spectrogram)
        _, predicted = torch.max(outputs, 1)
        for i,pred in enumerate(predicted):
            df.loc[df['file']==filename[i],'label']=int(pred.item())
df['label']=df['label'].astype(int)
df.to_csv('test.csv',index=False)