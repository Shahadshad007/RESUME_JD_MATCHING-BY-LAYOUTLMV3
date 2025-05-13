# Resume-JD Matching with LayoutLMv3

This project implements a resume and job description (JD) matching system using LayoutLMv3, a transformer-based model designed for document understanding. It processes resume PDFs, extracts text and layout information via OCR, and matches them against job descriptions to rank resumes based on relevance. The system leverages deep learning to understand both textual and spatial features of resumes, making it ideal for automated recruitment workflows.

## 🚀 Features

- **PDF to Image Conversion**: Converts resume PDFs to images for processing.  
- **OCR Extraction**: Uses Tesseract to extract text and bounding boxes from resume images.  
- **Data Preparation**: Cleans and formats resume and JD data into a structured dataset.  
- **Model Training**: Fine-tunes LayoutLMv3 for binary classification (match/no-match).  
- **Resume Ranking**: Scores and ranks resumes based on their match with a given JD.  
- **Output**: Generates a CSV with ranked resumes and match scores.  

## 🛠️ Installation

### Prerequisites

- Python 3.8+  
- CUDA-compatible GPU (optional, for faster training/inference)  
- Poppler (for PDF to image conversion)  

### Setup

**Clone the repository:**
```bash
git clone https://github.com/your-username/resume-jd-matching.git
cd resume-jd-matching
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Install Poppler:**

- **Windows**: Download and add Poppler to your PATH (e.g., via `conda install poppler`).  
- **Linux/Mac**: Install via package manager (e.g., `sudo apt-get install poppler-utils` or `brew install poppler`).  

**Download Tesseract:**

Install Tesseract OCR (`sudo apt-get install tesseract-ocr` on Linux or equivalent for your OS).  

## 📂 Project Structure

```
resume-jd-matching/
├── resume_pdfs_full/       # Input folder for resume PDFs
├── resume_images/          # Output folder for converted images
├── resume_ocr/             # Output folder for OCR JSON files
├── job_descriptions/       # Folder for JD text files
├── prepared_data.csv       # Cleaned dataset with resume/JD text and labels
├── layoutlmv3_resume_matching/  # Model training output
├── scripts/
│   ├── convert_pdfs.py     # Convert PDFs to images
│   ├── extract_ocr.py      # Extract OCR data
│   ├── prepare_data.py     # Prepare dataset
│   ├── train_model.py      # Train LayoutLMv3
│   ├── rank_resumes.py     # Rank resumes against a JD
├── requirements.txt        # Python dependencies
└── README.md
```

## 📝 Usage

### 1. Prepare Data

- Place resume PDFs in `resume_pdfs_full/`.  
- Place JD text files in `job_descriptions/` (e.g., `jd_0001.txt`).  
- Run the data preparation scripts:
```bash
python scripts/convert_pdfs.py
python scripts/extract_ocr.py
python scripts/prepare_data.py
```

### 2. Train the Model

Fine-tune the LayoutLMv3 model:
```bash
python scripts/train_model.py
```

The trained model is saved in `layoutlmv3_resume_matching/final_model/`.

### 3. Rank Resumes

Use a JD file (e.g., `software_developer.txt`) to rank resumes:
```bash
python scripts/rank_resumes.py
```

Results are saved in `resume_ranking_results.csv` with match scores and file paths.

## 📊 Example Output

**🏆 Top 5 Matching Resumes:**
```
--------------------------------------------------
1. resume_001.pdf
   Match Score: 0.9234
   Path: resume_pdfs_full/resume_001.pdf
   Pages: 1
--------------------------------------------------
...
```

## 🧠 Model Details

- **Base Model**: Microsoft LayoutLMv3-base  
- **Task**: Binary classification (match/no-match)  
- **Input**: Resume images, OCR data (text + bounding boxes), JD text  
- **Output**: Probability score for resume-JD match  
- **Training**: 3 epochs, batch size 2, learning rate 5e-5  

## ⚙️ Requirements

See `requirements.txt` for a full list. Key dependencies:

- `transformers`  
- `torch`  
- `pandas`  
- `pdf2image`  
- `pytesseract`  
- `PIL`  

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository.  
2. Create a feature branch (`git checkout -b feature/your-feature`).  
3. Commit changes (`git commit -m 'Add your feature'`).  
4. Push to the branch (`git push origin feature/your-feature`).  
5. Open a Pull Request.  


## 🙌 Acknowledgments

- Hugging Face Transformers for LayoutLMv3  
- Tesseract OCR for text extraction  
- pdf2image for PDF conversion  

---

Feel free to open an issue if you encounter any problems or have suggestions!  
**Happy matching! 🎉**
