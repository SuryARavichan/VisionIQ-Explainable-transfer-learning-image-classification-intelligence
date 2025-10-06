
from torch.utils.data import Dataset
from PIL import Image
import torch
import pandas as pd
from pathlib import Path


class ImageCsvDataset(Dataset):
def __init__(self, csv_file, transform=None):
self.df = pd.read_csv(csv_file)
self.transform = transform


def __len__(self):
return len(self.df)


def __getitem__(self, idx):
row = self.df.iloc[idx]
img = Image.open(row['filepath']).convert('RGB')
img = np.array(img)
if self.transform:
augmented = self.transform(image=img)
img = augmented['image']
label = int(row['label_idx'])
return img, torch.tensor(label, dtype=torch.long)




def save_checkpoint(model, path):
torch.save(model.state_dict(), str(path))




def accuracy(output, target):
preds = output.argmax(dim=1)
return (preds == target).float().mean().item()