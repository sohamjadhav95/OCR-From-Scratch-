# ✅ Phase 1: Data Preparation (For OCR with Neural Networks)

## Step: Convert Images to Preprocessed Tensors (.pt)

## ✅ What We Already Have
- Dataset with ~400,000 character images
- Organized into class folders (e.g., A, B, C or Sample001, Sample002, ...)
- Images are in JPG/PNG format and vary in size and color

## 🎯 What We're Doing in This Step
- Convert all images to **grayscale**
- Resize to fixed dimension (**32×32**)
- Normalize pixel values (0–255 → 0.0–1.0)
- Convert to **PyTorch tensors** with shape [1, 32, 32]
- Save each image as a `.pt` file
- Save a CSV mapping: `filename → label`

## 📂 Output Folder Structure

## 📥 Dataset

The training dataset (32×32 grayscale OCR images) can be downloaded from:

🔗 [Download Dataset from Google Drive](https://drive.google.com/file/d/1-r-htbRTrfOyIRhMFSkzz6S2XYYEgItq/view?usp=drive_link)

**Note**: After download, extract into `Data/ResizedDataset` before running training scripts.



# 📦 Phase 2: Dataset & DataLoader Setup (OCR Project)

## ✅ Objective
Prepare a PyTorch-compatible dataset loader that:
- Reads 400K+ grayscale images (32×32)
- Labels each image based on its folder
- Loads data in mini-batches
- Converts images to tensors for model training

---

## 📂 Dataset Structure

Each character class is stored in its own folder:

ProcessedDataset/
├── Sample001/ → Label 0 (Digit 0)
├── Sample002/ → Label 1 (Digit 1)
...
├── Sample011/ → Label 10 (A)
...
├── Sample062/ → Label 61 (z)


Total Classes: **62**  
(10 digits + 26 uppercase + 26 lowercase)

---

## 🧠 What We Did

1. **Created `OCRDataset` class** in `OCR_dataset.py`
2. Defined image transform:
   - Convert to grayscale (if needed)
   - Resize to 32×32 (safety step)
   - Normalize to [0, 1] float tensor
3. Extracted labels using folder index
4. Made compatible with `DataLoader`

---


### 🚀 Phase 3: Training Loop (`train.py`)

- Loads 32×32 grayscale images using custom `OCRDataset`
- Uses `OCRNet` CNN model for classification (62 output classes)
- Trains the model using `CrossEntropyLoss` and `Adam` optimizer
- Tracks training progress in real-time using 1–100% scale
- Saves final model weights to `ocr_model.pth`

