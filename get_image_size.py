from PIL import Image
import os, shutil

def get_num_pixels(filepath):
    width, height = Image.open(filepath).size
    return width, height

def get_pixels_proportion(filepath, dest_dir):
    is_square = False
    width, height = get_num_pixels(filepath)
    proportion = width / height

    if proportion <= 2: 
        is_square = True
        print(f"{filepath} --- Bu plaka kare plakadır")

        filename = os.path.basename(filepath)
        dest_path = os.path.join(dest_dir, filename)

        shutil.move(filepath, dest_path)

    return is_square


path = r"C:\Users\PC\Desktop\plates\detected_plates\07\31"
dest = r"C:\Users\PC\Desktop\plates\square_plates\07\square_plates_07_31"

square_plate_count = 0
with os.scandir(path) as files:
    for plate in files:
        if plate.is_file() and plate.name.endswith('.jpg'):
            is_square = get_pixels_proportion(plate.path, dest)
            if is_square:
                square_plate_count += 1

print(f"{square_plate_count}")
