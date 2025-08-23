import os
import torch
import torch.nn as nn
from PIL import Image, ImageOps
import torchvision.transforms as transforms

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return ImageOps.expand(image, padding, fill=0)

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

characters = '0123456789ABCDEFGHIJKLMNOPRSTUVYZ#'
idx_to_char = {idx+1: char for idx, char in enumerate(characters)}
idx_to_char[0] = ''  # blank = boş

transform = transforms.Compose([
    SquarePad(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def ctc_decode(preds):

    preds = preds.argmax(2)  # [T, N]
    preds = preds.cpu().numpy()
    results = []
    for col in preds.T:
        prev = -1
        text = ""
        for p in col:
            if p != prev and p != 0:
                text += idx_to_char[p]
            prev = p

        text = text.replace("#", "\n")
        results.append(text)
    return results

def predict_plate(model_path, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nclass = len(characters) + 1
    model = CRNN(128, 1, nclass, 256).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0).to(device)  # [1,1,H,W]

    with torch.no_grad():
        outputs = model(image)  # [T, 1, C]
        decoded = ctc_decode(outputs)

    return decoded[0]

if __name__ == "__main__":
    model_path = "turkish_plate_crnn_square_with_newline.pth"
    test_image = r"D:\Medias\plates\square_plates\square_plates_05_01\07ADM383.jpg"

    result = predict_plate(model_path, test_image)
    print("Tahmin:", result)
