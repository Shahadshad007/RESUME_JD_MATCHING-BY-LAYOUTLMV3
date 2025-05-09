import os
from pdf2image import convert_from_path

# Folder where resume PDFs are stored
pdf_folder = "resume_pdfs_full"
# Output folder where images will be saved
image_folder = "resume_images"
os.makedirs(image_folder, exist_ok=True)

# Loop through all PDF files
for idx, fname in enumerate(sorted(os.listdir(pdf_folder))):
    if not fname.endswith(".pdf"):
        continue

    pdf_path = os.path.join(pdf_folder, fname)
    # Convert PDF to images
    images = convert_from_path(pdf_path, dpi=300)  # 300 DPI for better quality

    # Save each page as PNG (one image per page)
    for page_num, image in enumerate(images):
        image.save(os.path.join(image_folder, f"resume_{idx:04d}_page_{page_num + 1}.png"), "PNG")

print(f"✅ All PDFs converted to images in '{image_folder}/'")
