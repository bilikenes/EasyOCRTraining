import os

klasor = r"D:\Medias\plates\square_plates\01"

for i in range(1, 32):
    klasor_adi = f"square_plates_01_{i:02}"
    klasor_yolu = os.path.join(klasor, klasor_adi)
    os.makedirs(klasor_yolu, exist_ok=True)

print("ok.")
