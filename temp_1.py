# import os

# klasor = r"C:\Users\PC\Desktop\plates\square_plates\07"

# for i in range(1, 32):
#     klasor_adi = f"square_plates_07_{i:02}"
#     klasor_yolu = os.path.join(klasor, klasor_adi)
#     os.makedirs(klasor_yolu, exist_ok=True)

# print("ok.")


import torch
from PIL import Image
import torchvision.transforms as transforms
from model import CRNN
from utils import beam_search_ctc
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN(128, 1, nclass=34, nh=256).to(device)
model.load_state_dict(torch.load("square_plate_crnn.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_folder = r"C:\Users\PC\Desktop\plates\square_plates\01\square_plates_01_31"

total = 0
correct = 0

for file_name in os.listdir(test_folder):
    if not file_name.lower().endswith(".jpg"):
        continue

    total += 1
    img_path = os.path.join(test_folder, file_name)
    img = Image.open(img_path).convert('L')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        outputs = outputs.softmax(2).squeeze(1)
        plate_pred = beam_search_ctc(outputs, beam_width=5)

    # Gerçek plaka (dosya adı)
    true_plate = os.path.splitext(file_name)[0].upper()
    success = (plate_pred == true_plate)
    if success:
        correct += 1

    print(f"{file_name} -> predict: {plate_pred}, real: {true_plate}, correct: {success}")

# Toplam doğruluk
accuracy = correct / total * 100
print(f"\nTotal images: {total}, Correct: {correct}, Accuracy: {accuracy:.2f}%")