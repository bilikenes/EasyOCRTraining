# import os
# import torch
# from PIL import Image
# import torchvision.transforms as transforms
# import torch.nn as nn

# class CRNN(nn.Module):
#     def __init__(self, imgH, nc, nclass, nh):
#         super(CRNN, self).__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(nc, 64, 3, 1, 1),
#             nn.ReLU(True),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(64, 128, 3, 1, 1),
#             nn.ReLU(True),
#             nn.MaxPool2d(2, 2)
#         )
#         self.rnn = nn.LSTM(128 * (imgH//4), nh, bidirectional=True, num_layers=2)
#         self.embedding = nn.Linear(nh*2, nclass)

#     def forward(self, x):
#         conv = self.cnn(x)
#         b, c, h, w = conv.size()
#         conv = conv.permute(3, 0, 2, 1)
#         conv = conv.reshape(w, b, -1)
#         rnn_out, _ = self.rnn(conv)
#         output = self.embedding(rnn_out)
#         return output

# characters = '0123456789ABCDEFGHIJKLMNOPRSTUVYZ'
# idx_to_char = {idx+1: char for idx, char in enumerate(characters)}
# idx_to_char[0] = ''  

# nclass = len(characters) + 1
# model = CRNN(32, 1, nclass, 256)
# model.load_state_dict(torch.load('turkish_plate_crnn.pth', map_location='cpu'))
# model.eval()

# transform = transforms.Compose([
#     transforms.Resize((32, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# def decode(preds):
#     prev = -1
#     result = ''
#     for p in preds:
#         if p != prev and p != 0:
#             result += idx_to_char[p.item()]
#         prev = p
#     return result


# test_folder = r'D:\Medias\plates\detected_plates' 
# results = []

# for file_name in os.listdir(test_folder):
#     if file_name.lower().endswith('.jpg'):
#         img_path = os.path.join(test_folder, file_name)
#         img = Image.open(img_path).convert('L')
#         img = transform(img).unsqueeze(0)  # batch dimension
#         with torch.no_grad():
#             outputs = model(img)
#             outputs = outputs.softmax(2)
#             preds = outputs.argmax(2).squeeze(1)
#             plate = decode(preds)
#         results.append((file_name, plate))
#         print(f"{file_name} -> {plate}")

# import pandas as pd
# df = pd.DataFrame(results, columns=['image', 'predicted_plate'])
# df.to_csv('predictions.csv', index=False)
# print("Tüm tahminler kaydedildi: predictions.csv")

import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import pandas as pd

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
        conv = conv.permute(3, 0, 2, 1)
        conv = conv.reshape(w, b, -1)
        rnn_out, _ = self.rnn(conv)
        output = self.embedding(rnn_out)
        return output

characters = '0123456789ABCDEFGHIJKLMNOPRSTUVYZ'
idx_to_char = {idx+1: char for idx, char in enumerate(characters)}
idx_to_char[0] = ''  # CTC blank

nclass = len(characters) + 1
model = CRNN(32, 1, nclass, 256)
model.load_state_dict(torch.load('turkish_plate_crnn.pth', map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def decode(preds):
    prev = -1
    result = ''
    for p in preds:
        if p != prev and p != 0:
            result += idx_to_char[p.item()]
        prev = p
    return result

test_folder =  r'D:\Medias\plates\detected_plates\detected_plates_03_01' 
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

        true_plate = os.path.splitext(file_name)[0].upper()
        success = (plate == true_plate)
        if success:
            correct += 1

        results.append((file_name, plate, true_plate, success))
        print(f"{file_name} -> predict: {plate}, real: {true_plate}, result: {success}")

accuracy = correct / total * 100
print(f"\ntotal: {total}, true: {correct}, accuracy: {accuracy:.2f}%")


df = pd.DataFrame(results, columns=['image', 'predicted_plate', 'true_plate', 'success'])
df.to_csv('predictions_with_accuracy.csv', index=False)
print("ok : predictions_with_accuracy.csv")
