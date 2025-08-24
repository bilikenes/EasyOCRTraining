import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import pandas as pd

# Model tanımı (kare plaka CRNN aynı yapı)
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


# Karakterler ve index eşlemeleri
characters = '0123456789ABCDEFGHIJKLMNOPRSTUVYZ'
idx_to_char = {idx+1: char for idx, char in enumerate(characters)}
idx_to_char[0] = ''  # CTC blank

nclass = len(characters) + 1

# Kare plaka modeli yükle
model = CRNN(128, 1, nclass, 256)  # dikkat: yükseklik 128
model.load_state_dict(torch.load('square_plate_crnn.pth', map_location='cpu'))
model.eval()

# Transform (kare plakaya göre)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# CTC decode
def decode(preds):
    prev = -1
    result = ''
    for p in preds:
        if p != prev and p != 0:
            result += idx_to_char[p.item()]
        prev = p
    return result

test_folder = r'C:\Users\PC\Desktop\plates\square_plates\01\square_plates_01_30'  
results = []
correct = 0
total = 0

for file_name in os.listdir(test_folder):
    if file_name.lower().endswith('.jpg'):
        total += 1
        img_path = os.path.join(test_folder, file_name)
        img = Image.open(img_path).convert('L')
        img = transform(img).unsqueeze(0)  # batch dimension

        with torch.no_grad():
            outputs = model(img)
            outputs = outputs.softmax(2)
            preds = outputs.argmax(2).squeeze(1)
            plate = decode(preds)

        # Gerçek label -> dosya adı
        true_plate = os.path.splitext(file_name)[0].upper()
        success = (plate == true_plate)
        if success:
            correct += 1

        results.append((file_name, plate, true_plate, success))
        print(f"{file_name} -> predict: {plate}, real: {true_plate}, result: {success}")

# Accuracy hesapla
accuracy = correct / total * 100
print(f"\ntotal: {total}, true: {correct}, accuracy: {accuracy:.2f}%")

# CSV olarak kaydet
df = pd.DataFrame(results, columns=['image', 'predicted_plate', 'true_plate', 'success'])
df.to_csv('square_plate_predictions.csv', index=False)
print("ok : square_plate_predictions.csv")
