import os
import pytesseract
from pytesseract import Output
from PIL import Image
import json

# Folder containing the resume images
image_folder = "image_path"
# Output folder for OCR data
ocr_folder = "folder_for_output"
os.makedirs(ocr_folder, exist_ok=True)

# Loop through all images
for idx, fname in enumerate(sorted(os.listdir(image_folder))):
    if not fname.endswith(".png"):
        continue

    image_path = os.path.join(image_folder, fname)
    # Open image
    img = Image.open(image_path)

    # Run OCR on the image
    ocr_data = pytesseract.image_to_data(img, output_type=Output.DICT)

    # Collect text and bounding boxes
    ocr_output = []
    n = len(ocr_data["level"])
    for i in range(n):
        text = ocr_data["text"][i].strip()
        if not text:
            continue
        x, y, w, h = ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i]
        ocr_output.append({
            "token": text,
            "bbox": [x, y, x + w, y + h]
        })

    # Save OCR result as JSON
    json_path = os.path.join(ocr_folder, f"{os.path.splitext(fname)[0]}.json")
    with open(json_path, "w") as json_file:
        json.dump(ocr_output, json_file)

print(f"✅ OCR and bounding boxes extracted and saved in '{ocr_folder}/'")
