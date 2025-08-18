import os
import csv

folder_path = r"D:\Medias\plates\detected_plates_07_31"

csv_path = r"D:\Medias\plates\detected_plates_07_31\labels.csv"

with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['filename', 'plate'])  
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.jpg'):
            plate_name = os.path.splitext(filename)[0] 
            writer.writerow([filename, plate_name])

print(f"ok: {csv_path}")
