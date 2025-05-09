# ------------------------------
# Resume & JD Matching - Final Training Script
# Using LayoutLMv3 (with built-in OCR)
# ------------------------------

# Set specific GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 📦 Imports
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from PIL import Image
from sklearn.model_selection import train_test_split
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score
import numpy as np

# ------------------------------
# 1. Dataset Definition
# ------------------------------

class ResumeJDDataset(Dataset):
    def __init__(self, df, image_folder, jd_folder, processor):
        self.df = df.reset_index(drop=True)
        self.image_folder = image_folder
        self.jd_folder = jd_folder
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]

        image_path = os.path.join(self.image_folder, f"resume_{idx:04d}_page_1.png")
        jd_path = os.path.join(self.jd_folder, f"jd_{idx:04d}.txt")

        image = Image.open(image_path).convert("RGB")
        with open(jd_path, "r", encoding="utf-8") as f:
            jd_text = f.read()

        encoded = self.processor(
            images=image,
            text=jd_text,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        encoded["labels"] = torch.tensor(row["label"], dtype=torch.long)
        return encoded

# ------------------------------
# 2. Load Processor & Data
# ------------------------------

processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
df = pd.read_csv("CSV_FILE")
dataset = ResumeJDDataset(df, "resume_images", "job_descriptions", processor)
print(f"✅ Dataset ready with {len(dataset)} samples.")

# ------------------------------
# 3. Train / Eval Split
# ------------------------------

train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
train_dataset = Subset(dataset, train_idx)
eval_dataset = Subset(dataset, val_idx)
print(f"✅ Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

# ------------------------------
# 4. Model Setup
# ------------------------------

model = LayoutLMv3ForSequenceClassification.from_pretrained(
    "microsoft/layoutlmv3-base", num_labels=2
)

# ------------------------------
# 5. Training Arguments (No Evaluation During Training)
# ------------------------------

training_args = TrainingArguments(
    output_dir="./layoutlmv3_resume_matching",
    evaluation_strategy="no",       
    save_strategy="epoch",         # Save only once per epoch to reduce I/O
    logging_strategy="steps",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    remove_unused_columns=False,
    push_to_hub=False,
)

# ------------------------------
# 6. Trainer Setup
# ------------------------------

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # used only for post-training evaluation
    tokenizer=processor,
    compute_metrics=compute_metrics
)

# ------------------------------
# 7. Train the Model
# ------------------------------

trainer.train()

# ------------------------------
# 8. Post-Training Evaluation
# ------------------------------

metrics = trainer.evaluate()
print(f"📊 Final Evaluation Accuracy: {metrics['eval_accuracy']:.4f}")

# ------------------------------
# 9. Save Final Model
# ------------------------------

model.save_pretrained("PATH_TO_SAVE_MODEL")
processor.save_pretrained("PATH_TO_SAVE_MODE")

print("🎯 Training Completed Successfully and Model Saved!")
