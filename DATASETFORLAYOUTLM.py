import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import LayoutLMv3Processor


class ResumeJDDataset(Dataset):
    def __init__(self, df, image_folder, ocr_folder, jd_folder, processor):
        self.df = df.reset_index(drop=True)
        self.image_folder = Path(image_folder)
        self.ocr_folder = Path(ocr_folder)
        self.jd_folder = Path(jd_folder)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        base = f"resume_{idx:04d}_page_1"

        # Load image
        image_path = self.image_folder / f"{base}.png"
        image = Image.open(image_path).convert("RGB")

        # Load OCR
        ocr_path = self.ocr_folder / f"{base}.json"
        with open(ocr_path) as f:
            ocr_data = json.load(f)

        words = [item["token"] for item in ocr_data]
        boxes = [item["bbox"] for item in ocr_data]

        # Normalize bbox to 0-1000 as LayoutLM expects
        width, height = image.size
        norm_boxes = []
        for bbox in boxes:
            x0, y0, x1, y1 = bbox
            norm_box = [
                int(1000 * x0 / width),
                int(1000 * y0 / height),
                int(1000 * x1 / width),
                int(1000 * y1 / height)
            ]
            norm_boxes.append(norm_box)

        # Load JD text
        jd_path = self.jd_folder / f"jd_{idx:04d}.txt"
        with open(jd_path, "r") as f:
            jd_text = f.read()

        # Process everything
        encoded = self.processor(
            images=image,
            words=words,
            boxes=norm_boxes,
            text=jd_text,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        item = {k: v.squeeze() for k, v in encoded.items()}
        item["labels"] = torch.tensor(row["label"], dtype=torch.long)
        return item


# Define paths
path_to_csv = Path("prepared_data.csv")
path_to_image_folder = Path("resume_images")
path_to_ocr_folder = Path("resume_ocr")
path_to_jd_folder = Path("job_descriptions")

# Setup the processor and dataset
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")

df = pd.read_csv(path_to_csv)  # Load your final prepared data

dataset = ResumeJDDataset(
    df=df,
    image_folder=path_to_image_folder,
    ocr_folder=path_to_ocr_folder,
    jd_folder=path_to_jd_folder,
    processor=processor
)

print(f"✅ ResumeJDDataset ready with {len(dataset)} samples")
