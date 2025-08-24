import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class PlateDataset(Dataset):
    def __init__(self, csv_files, img_dirs, transform=None):
        self.data = []
        self.transform = transform

        for csv_file, img_dir in zip(csv_files, img_dirs):
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                img_path = os.path.join(img_dir, row['filename'])  # <-- CSV'deki resim kolonu
                label = row['plate']  # <-- CSV'deki plate kolonu
                if pd.notna(label):   # boş değerleri atla
                    self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('L')  # griye çevir

        if self.transform:
            img = self.transform(img)

        # karakterleri indexe çevir
        characters = '0123456789ABCDEFGHIJKLMNOPRSTUVYZ'
        char_to_idx = {char: i+1 for i, char in enumerate(characters)}  # 0 = CTC blank
        label_encoded = [char_to_idx[c] for c in label if c in char_to_idx]

        return img, label_encoded
