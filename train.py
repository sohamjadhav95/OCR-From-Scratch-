import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from OCR_Dataset import OCRDataset
from model import OCRNet
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt


EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
DATA_DIR = r"D:\Project Backups\OCR-From-Scratch-\EMNIST Dataset 80000+ Samples\Main - Copy"
MODEL_SAVE_PATH = "ocr_model_best.pth"
LABEL_MAP_PATH = "label_map.json"
VAL_SPLIT = 0.1
EARLY_STOPPING_PATIENCE = 5

if __name__ == "__main__":
    dataset = OCRDataset(root_dir=DATA_DIR)
    train_size = int((1 - VAL_SPLIT) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OCRNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    label_map = {i: str(i) for i in range(0, 10)}
    label_map.update({i: chr(ord('A') + (i - 10)) for i in range(10, 36)})
    label_map.update({i: chr(ord('a') + (i - 36)) for i in range(36, 62)})

    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f)

    best_val_acc = 0.0
    no_improve_count = 0

    # For plotting
    train_acc_list, val_acc_list = [], []
    train_loss_list, val_loss_list = [], []

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

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
        train_loss = running_loss / len(train_loader)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)

        model.eval()
        val_correct, val_total, val_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation", unit="batch"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss = val_loss_sum / len(val_loader)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

        scheduler.step(val_loss)

        print(f"✅ Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"💾 Best model saved (Val Acc = {best_val_acc:.2f}%)")
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= EARLY_STOPPING_PATIENCE:
                print("⏹ Early stopping triggered.")
                break

    print(f"\n🎉 Training complete! Best model: {MODEL_SAVE_PATH} | Best Val Acc = {best_val_acc:.2f}%")

    # 📊 Plot training curves
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_acc_list, label="Train Acc")
    plt.plot(val_acc_list, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Accuracy over Epochs")

    plt.subplot(1, 2, 2)
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(val_loss_list, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over Epochs")

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()
