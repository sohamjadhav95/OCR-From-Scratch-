import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from OCR_Dataset import OCRDataset
from model import OCRNet
import os
import json
from tqdm import tqdm

# ----------------------
# 🔧 Configurations
# ----------------------
EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DATA_DIR = r"E:\Projects\Master Projects (Core)\OCR-From-Scratch-\Data\grascale+processed+resized"
MODEL_SAVE_PATH = "ocr_model_best.pth"
LABEL_MAP_PATH = "label_map.json"
VAL_SPLIT = 0.1  # 10% validation

if __name__ == "__main__":
    # ----------------------
    # 📦 Load dataset and split
    # ----------------------
    dataset = OCRDataset(root_dir=DATA_DIR)
    train_size = int((1 - VAL_SPLIT) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # ----------------------
    # 🧠 Model, Loss, Optimizer
    # ----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OCRNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    # ----------------------
    # 💾 Create and Save Label Map
    # ----------------------
    label_map = {}
    for i in range(0, 10):
        label_map[i] = str(i)  # 0-9
    for i in range(10, 36):
        label_map[i] = chr(ord('A') + (i - 10))  # A-Z
    for i in range(36, 62):
        label_map[i] = chr(ord('a') + (i - 36))  # a-z

    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f)
    print(f"📄 Label map saved to {LABEL_MAP_PATH}")

    # ----------------------
    # 🔁 Training Loop
    # ----------------------
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"\n🌀 Epoch {epoch+1}/{EPOCHS}")
        for images, labels in tqdm(train_loader, desc="Training", unit="batch"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total

        # ----------------------
        # 📊 Validation
        # ----------------------
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation", unit="batch"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        print(f"✅ Epoch {epoch+1} Complete: Train Loss = {running_loss/len(train_loader):.4f} | Train Acc = {train_acc:.2f}% | Val Loss = {val_loss/len(val_loader):.4f} | Val Acc = {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"💾 Best model saved with Val Acc = {best_val_acc:.2f}%")

    print(f"\n🎉 Training complete! Best model saved to: {MODEL_SAVE_PATH}")
