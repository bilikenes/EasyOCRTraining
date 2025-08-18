import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence

class PlateDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.characters = '0123456789ABCDEFGHIJKLMNOPRSTUVYZ'
        self.char_to_idx = {char: idx+1 for idx, char in enumerate(self.characters)}
        self.char_to_idx['blank'] = 0  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('L')  # gri ton
        label_str = self.data.iloc[idx, 1]
        label = [self.char_to_idx[c] for c in label_str]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = PlateDataset(csv_file='D:/Medias/dataset_for_OCR/images/labels.csv', img_dir='D:/Medias/dataset_for_OCR/images', transform=transform)
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
model = CRNN(32, 1, nclass, 256)

ctc_loss = nn.CTCLoss(blank=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(150):  
    model.train()
    total_loss = 0
    for batch in dataloader:
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
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

torch.save(model.state_dict(), 'turkish_plate_crnn.pth')
print("ok")
