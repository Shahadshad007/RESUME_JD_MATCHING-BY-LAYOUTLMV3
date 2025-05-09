import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from pdf2image import convert_from_path
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForSequenceClassification
)


def main():
    # PATHS - MODIFY THESE TO MATCH YOUR ENVIRONMENT
    path_to_jd = Path("software_developer.txt")  # Path to the job description text file
    path_to_resume_folder = Path("resume_pdfs_full")  # Folder containing resume PDFs
    path_to_model = Path("./layoutlmv3_resume_matching/final_model")  # Trained model path
    path_to_output_csv = Path("resume_ranking_results.csv")  # Output CSV path

    top_k = 25  # Number of top matches to return
    gpu_id = "0"  # GPU ID to use

    # Set GPU if available
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and processor
    print("Loading model and processor...")
    processor = LayoutLMv3Processor.from_pretrained(path_to_model)
    model = LayoutLMv3ForSequenceClassification.from_pretrained(path_to_model)
    model.to(device)
    model.eval()

    # Load job description
    print(f"Reading job description from {path_to_jd}")
    with open(path_to_jd, "r", encoding="utf-8") as f:
        job_description = f.read()

    # Get all resume PDF files
    valid_extensions = [".pdf"]
    resume_files = [
        f for f in os.listdir(path_to_resume_folder)
        if (path_to_resume_folder / f).is_file() and
        any(f.lower().endswith(ext) for ext in valid_extensions)
    ]
    print(f"Found {len(resume_files)} resume PDFs to process")

    # Process each resume and compute match score
    results = []

    for resume_file in tqdm(resume_files, desc="Processing resumes"):
        resume_path = path_to_resume_folder / resume_file

        try:
            # Convert PDF to images (one image per page)
            images = convert_from_path(resume_path)
            page_scores = []

            # Process each page of the PDF
            for page_num, image in enumerate(images):
                image = image.convert("RGB")  # Ensure image is in RGB format

                encoded = processor(
                    images=image,
                    text=job_description,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )

                # Move inputs to device
                encoded = {k: v.to(device) for k, v in encoded.items()}

                # Get prediction
                with torch.no_grad():
                    outputs = model(**encoded)
                    logits = outputs.logits.cpu().numpy()[0]

                # Convert logits to probabilities
                probabilities = softmax(logits)
                match_score = probabilities[1]  # Class 1 probability (match)
                page_scores.append(match_score)

            # Aggregate scores (e.g., average across pages)
            avg_score = np.mean(page_scores) if page_scores else 0.0

            results.append({
                "resume_file": resume_file,
                "match_score": float(avg_score),
                "path": str(resume_path),
                "num_pages": len(images)
            })

        except Exception as e:
            print(f"Error processing {resume_file}: {str(e)}")

    # Sort results by match score (descending)
    results.sort(key=lambda x: x["match_score"], reverse=True)

    # Display top K results
    print(f"\n🏆 Top {top_k} Matching Resumes:")
    print("-" * 50)

    top_results = results[:top_k]
    for i, result in enumerate(top_results, 1):
        print(f"{i}. {result['resume_file']}")
        print(f"   Match Score: {result['match_score']:.4f}")
        print(f"   Path: {result['path']}")
        print(f"   Pages: {result['num_pages']}")
        print("-" * 50)

    # Save results to CSV
    output_df = pd.DataFrame(results)
    output_df.to_csv(path_to_output_csv, index=False)
    print(f"Full results saved to {path_to_output_csv}")


def softmax(x):
    """Compute softmax values for array of logits."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


if __name__ == "__main__":
    main()
