import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import PlateDataset
from model import CRNN

# Karakter seti
characters = '0123456789ABCDEFGHIJKLMNOPRSTUVYZ'
nclass = len(characters) + 1

# Transform (hafif augmentasyon)
train_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.RandomAffine(degrees=3, translate=(0.05,0.05), scale=(0.95,1.05)),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

csv_files = [
    r"C:\Users\PC\Desktop\plates\square_plates\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_01\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_02\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_03\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_04\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_05\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_06\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_07\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_08\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_09\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_10\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_11\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_12\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_13\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_14\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_15\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_16\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_17\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_18\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_19\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_20\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_21\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_22\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_23\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_24\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_25\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_26\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_27\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_28\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_29\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_30\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_31\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\05\square_plates_05_01\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\03\square_plates_03_01\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\03\square_plates_03_02\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\03\square_plates_03_03\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\03\square_plates_03_04\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\03\square_plates_03_05\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\03\square_plates_03_06\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\03\square_plates_03_07\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\03\square_plates_03_08\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\03\square_plates_03_09\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\03\square_plates_03_10\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_01\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_02\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_03\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_04\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_05\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_06\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_01\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_08\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_09\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_10\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_11\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_12\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_13\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_14\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_15\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_16\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_17\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_18\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_19\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_20\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_21\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_22\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_23\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_24\labels.csv",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_25\labels.csv"
]

img_dirs = [
    r"C:\Users\PC\Desktop\plates\square_plates",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_01",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_02",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_03",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_04",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_05",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_06",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_07",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_08",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_09",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_10",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_11",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_12",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_13",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_14",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_15",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_16",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_17",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_18",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_19",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_20",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_21",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_22",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_23",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_24",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_25",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_26",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_27",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_28",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_29",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_30",
    r"C:\Users\PC\Desktop\plates\square_plates_old\07\square_plates_07_31",
    r"C:\Users\PC\Desktop\plates\square_plates_old\05\square_plates_05_01",
    r"C:\Users\PC\Desktop\plates\square_plates_old\03\square_plates_03_01",
    r"C:\Users\PC\Desktop\plates\square_plates_old\03\square_plates_03_02",
    r"C:\Users\PC\Desktop\plates\square_plates_old\03\square_plates_03_03",
    r"C:\Users\PC\Desktop\plates\square_plates_old\03\square_plates_03_04",
    r"C:\Users\PC\Desktop\plates\square_plates_old\03\square_plates_03_05",
    r"C:\Users\PC\Desktop\plates\square_plates_old\03\square_plates_03_06",
    r"C:\Users\PC\Desktop\plates\square_plates_old\03\square_plates_03_07",
    r"C:\Users\PC\Desktop\plates\square_plates_old\03\square_plates_03_08",
    r"C:\Users\PC\Desktop\plates\square_plates_old\03\square_plates_03_09",
    r"C:\Users\PC\Desktop\plates\square_plates_old\03\square_plates_03_10",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_01",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_02",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_03",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_04",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_05",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_06",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_01",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_08",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_09",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_10",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_11",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_12",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_13",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_14",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_15",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_16",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_17",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_18",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_19",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_20",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_21",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_22",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_23",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_24",
    r"C:\Users\PC\Desktop\plates\square_plates_old\01\square_plates_01_25"

]

# Dataset ve DataLoader
dataset = PlateDataset(csv_files=csv_files, img_dirs=img_dirs, transform=train_transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=lambda x: x)

# Daha derin CRNN örneği
class DeepCRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,1,1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,1,1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2,1), (2,1)),  # sadece height azalt
            nn.Conv2d(256,256,3,1,1), nn.BatchNorm2d(256), nn.ReLU(True)
        )
        self.rnn = nn.LSTM(256* (imgH//8), nh, bidirectional=True, num_layers=2)
        self.embedding = nn.Linear(nh*2, nclass)

    def forward(self,x):
        conv = self.cnn(x)
        b,c,h,w = conv.size()
        conv = conv.permute(3,0,2,1).reshape(w,b,-1)
        rnn_out,_ = self.rnn(conv)
        output = self.embedding(rnn_out)
        return output

# Model ve optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepCRNN(128,1,nclass,256).to(device)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

# Eğitim döngüsü
epochs = 10
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        imgs, targets, target_lengths = [], [], []
        for sample in batch:
            img, label = sample
            imgs.append(img)
            targets.extend(label)
            target_lengths.append(len(label))

        imgs = torch.stack(imgs).to(device)
        targets = torch.tensor(targets, dtype=torch.long).to(device)
        input_lengths = torch.full((len(batch),), imgs.size(3), dtype=torch.long).to(device)

        optimizer.zero_grad()
        outputs = model(imgs)  # [T, B, C]
        T, B, C = outputs.size()
        input_lengths = torch.full((B,), T, dtype=torch.long).to(device)
        loss = criterion(outputs.log_softmax(2), targets, input_lengths, torch.tensor(target_lengths).to(device))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        epoch_loss += loss.item()

    scheduler.step()
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss/len(dataloader):.4f}")

torch.save(model.state_dict(),"deep_square_plate_crnn.pth")
