import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm

class PlateDataset(Dataset):
    def __init__(self, csv_files, img_dirs, transform=None):
        all_data = []
        for csv_file, img_dir in zip(csv_files, img_dirs):
            df = pd.read_csv(csv_file)
            df['img_dir'] = img_dir
            all_data.append(df)

        self.data = pd.concat(all_data, ignore_index=True)
        self.transform = transform
        self.characters = '0123456789ABCDEFGHIJKLMNOPRSTUVYZ'
        self.char_to_idx = {char: idx+1 for idx, char in enumerate(self.characters)}
        self.char_to_idx['blank'] = 0 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data.iloc[idx]['img_dir'], self.data.iloc[idx, 0])
        image = Image.open(img_path).convert('L') 
        label_str = self.data.iloc[idx, 1]
        label = [self.char_to_idx[c] for c in label_str]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

transform = transforms.Compose([
    transforms.Resize((64, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

csv_files = [
    r"D:\Medias\plates\detected_plates\detected_plates_07_01\labels.csv",
    r"D:\Medias\plates\detected_plates\detected_plates_07_02\labels.csv",
    r"D:\Medias\plates\detected_plates\detected_plates_07_03\labels.csv",
    r"D:\Medias\plates\detected_plates\detected_plates_07_04\labels.csv",
    r"D:\Medias\plates\detected_plates\detected_plates_07_05\labels.csv",
    r"D:\Medias\plates\detected_plates\detected_plates_07_06\labels.csv",
    r"D:\Medias\plates\detected_plates\detected_plates_07_07\labels.csv",
    r"D:\Medias\plates\detected_plates\detected_plates_07_08\labels.csv",
    r"D:\Medias\plates\detected_plates\detected_plates_07_09\labels.csv",
    r"D:\Medias\plates\detected_plates\detected_plates_07_10\labels.csv",
    r"D:\Medias\plates\detected_plates\detected_plates_07_11\labels.csv",
    r"D:\Medias\plates\detected_plates\detected_plates_07_12\labels.csv",
    r"D:\Medias\plates\detected_plates\detected_plates_07_13\labels.csv",
    r"D:\Medias\plates\detected_plates\detected_plates_07_14\labels.csv",
    r"D:\Medias\plates\detected_plates\detected_plates_07_15\labels.csv",
    r"D:\Medias\plates\detected_plates\detected_plates_07_16\labels.csv",
    r"D:\Medias\plates\detected_plates\detected_plates_07_17\labels.csv",
    r"D:\Medias\plates\detected_plates\detected_plates_07_18\labels.csv",
    r"D:\Medias\plates\detected_plates\detected_plates_07_19\labels.csv",
    r"D:\Medias\plates\detected_plates\detected_plates_07_20\labels.csv",
    r"D:\Medias\plates\square_plates\square_plates_03_01\labels.csv",
    r"D:\Medias\plates\square_plates\square_plates_03_02\labels.csv",
    r"D:\Medias\plates\square_plates\square_plates_03_03\labels.csv",
    r"D:\Medias\plates\square_plates\square_plates_03_04\labels.csv",
    r"D:\Medias\plates\square_plates\square_plates_03_05\labels.csv",
    r"D:\Medias\plates\square_plates\square_plates_03_06\labels.csv",
    r"D:\Medias\plates\square_plates\square_plates_03_07\labels.csv",
    r"D:\Medias\plates\square_plates\square_plates_03_08\labels.csv",
    r"D:\Medias\plates\square_plates\square_plates_03_09\labels.csv",
    r"D:\Medias\plates\square_plates\square_plates_03_10\labels.csv"
]

img_dirs = [
    r"D:\Medias\plates\detected_plates\detected_plates_07_01",
    r"D:\Medias\plates\detected_plates\detected_plates_07_02",
    r"D:\Medias\plates\detected_plates\detected_plates_07_03",
    r"D:\Medias\plates\detected_plates\detected_plates_07_04",
    r"D:\Medias\plates\detected_plates\detected_plates_07_05",
    r"D:\Medias\plates\detected_plates\detected_plates_07_06",
    r"D:\Medias\plates\detected_plates\detected_plates_07_07",
    r"D:\Medias\plates\detected_plates\detected_plates_07_08",
    r"D:\Medias\plates\detected_plates\detected_plates_07_09",
    r"D:\Medias\plates\detected_plates\detected_plates_07_10",
    r"D:\Medias\plates\detected_plates\detected_plates_07_11",
    r"D:\Medias\plates\detected_plates\detected_plates_07_12",
    r"D:\Medias\plates\detected_plates\detected_plates_07_13",
    r"D:\Medias\plates\detected_plates\detected_plates_07_14",
    r"D:\Medias\plates\detected_plates\detected_plates_07_15",
    r"D:\Medias\plates\detected_plates\detected_plates_07_16",
    r"D:\Medias\plates\detected_plates\detected_plates_07_17",
    r"D:\Medias\plates\detected_plates\detected_plates_07_18",
    r"D:\Medias\plates\detected_plates\detected_plates_07_19",
    r"D:\Medias\plates\detected_plates\detected_plates_07_20",
    r"D:\Medias\plates\square_plates\square_plates_03_01",
    r"D:\Medias\plates\square_plates\square_plates_03_02",
    r"D:\Medias\plates\square_plates\square_plates_03_03",
    r"D:\Medias\plates\square_plates\square_plates_03_04",
    r"D:\Medias\plates\square_plates\square_plates_03_05",
    r"D:\Medias\plates\square_plates\square_plates_03_06",
    r"D:\Medias\plates\square_plates\square_plates_03_07",
    r"D:\Medias\plates\square_plates\square_plates_03_08",
    r"D:\Medias\plates\square_plates\square_plates_03_09",
    r"D:\Medias\plates\square_plates\square_plates_03_10"
]

dataset = PlateDataset(csv_files=csv_files, img_dirs=img_dirs, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=lambda x: x)

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.rnn = nn.LSTM(128 * (imgH//4), nh, bidirectional=True, num_layers=2)
        self.embedding = nn.Linear(nh*2, nclass)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.permute(3, 0, 2, 1)  # [w, b, h, c]
        conv = conv.reshape(w, b, -1)
        rnn_out, _ = self.rnn(conv)
        output = self.embedding(rnn_out)
        return output

nclass = len(dataset.characters) + 1
model = CRNN(64, 1, nclass, 256)


ctc_loss = nn.CTCLoss(blank=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(10):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", unit="batch")
    
    for batch in progress_bar:
        images, labels = zip(*batch)
        images = torch.stack(images).to(device)
        labels_flat = torch.cat(labels).to(device)
        label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
        
        optimizer.zero_grad()
        outputs = model(images)
        seq_len = torch.full(size=(images.size(0),), fill_value=outputs.size(0), dtype=torch.long)
        loss = ctc_loss(outputs.log_softmax(2), labels_flat, seq_len, label_lengths)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    print(f"Epoch {epoch+1} Completed, Avg Loss: {total_loss/len(dataloader):.4f}")

torch.save(model.state_dict(), 'turkish_plate_crnn.pth')
print("ok")