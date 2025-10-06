# data_prep.py
from pathlib import Path
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 42


def seed_everything(seed=SEED):
random.seed(seed)
np.random.seed(seed)




def prepare_dataset(image_dir: str, labels_csv: str, out_dir: str = "data/processed", val_size: float = 0.2, test_size: float = 0.1):
"""
Reads a labels CSV with columns [image, label], validates files, creates train/val/test splits, and writes csv manifests.
"""
seed_everything()
root = Path(image_dir)
df = pd.read_csv(labels_csv)
# Validate path
df['filepath'] = df['image'].apply(lambda p: str(root / p))
df = df[df['filepath'].apply(lambda p: Path(p).exists())].copy()


# map labels to integers
labels = sorted(df['label'].unique())
label2idx = {l: i for i, l in enumerate(labels)}
df['label_idx'] = df['label'].map(label2idx)


# first split off test
trainval, test = train_test_split(df, test_size=test_size, stratify=df['label_idx'], random_state=SEED)
train, val = train_test_split(trainval, test_size=val_size/(1-test_size), stratify=trainval['label_idx'], random_state=SEED)


out = Path(out_dir)
out.mkdir(parents=True, exist_ok=True)
train.to_csv(out / 'train.csv', index=False)
val.to_csv(out / 'val.csv', index=False)
test.to_csv(out / 'test.csv', index=False)
pd.Series(labels).to_csv(out / 'labels.txt', index=False, header=False)


return {
'train_csv': str(out / 'train.csv'),
'val_csv': str(out / 'val.csv'),
'test_csv': str(out / 'test.csv'),
'labels': labels
}


if __name__ == '__main__':
# quick CLI usage
import argparse
p = argparse.ArgumentParser()
p.add_argument('--image_dir', required=True)
p.add_argument('--labels_csv', required=True)
p.add_argument('--out_dir', default='data/processed')
args = p.parse_args()
print(prepare_dataset(args.image_dir, args.labels_csv, args.out_dir))