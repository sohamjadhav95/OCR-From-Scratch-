import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from OCR_Dataset import OCRDataset
from model import OCRNet
import os

# ----------------------
# 🔧 Configurations
# ----------------------
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DATA_DIR = r"E:\Projects\Master Projects (Core)\OCR-From-Scratch-\Data\grascale+processed+resized"
MODEL_SAVE_PATH = "ocr_model.pth"

# For multiprocessing and Erroless training On WINDOWS Wrap All training in if __name__ == "__main__":

if __name__ == "__main__":
    # ----------------------
    # 📦 Load dataset
    # ----------------------
    dataset = OCRDataset(root_dir=DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # ----------------------
    # 🧠 Model, Loss, Optimizer
    # ----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OCRNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ----------------------
    # 🔁 Training Loop
    # ----------------------
    total_steps = EPOCHS * len(dataloader)
    step = 0

    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(dataloader):
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

            step += 1
            progress = int((step / total_steps) * 100)
            print(f"\r🌀 Progress: {progress}% | Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f}", end='')

        accuracy = 100 * correct / total
        print(f"\n✅ Epoch {epoch+1} Complete: Loss = {running_loss/len(dataloader):.4f} | Accuracy = {accuracy:.2f}%")

    # ----------------------
    # 💾 Save Trained Model
    # ----------------------
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n🎉 Training complete! Model saved to: {MODEL_SAVE_PATH}")
