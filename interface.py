import torch
from torchvision import transforms
from PIL import Image
import json
from model import OCRNet

MODEL_PATH = "D:\Project Backups\OCR-From-Scratch-\Models\90%\ocr_model_best (1).pth"
LABEL_MAP_PATH = "D:\Project Backups\OCR-From-Scratch-\Models\90%\label_map.json"
IMAGE_PATH = r"C:\Users\soham\OneDrive\Pictures\Screenshots\Screenshot 2025-08-21 141948.png"  # change this

# Load label mapping
with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)


# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OCRNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)  # <--- MOVE MODEL TO DEVICE
model.eval()


# Preprocessing (same as training)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image.to(device))
        _, predicted = torch.max(outputs, 1)
        predicted_idx = str(predicted.item())  # convert to str for json keys
    return label_map[predicted_idx]

if __name__ == "__main__":
    prediction = predict_image(IMAGE_PATH)
    print(f"Predicted Character: {prediction}")
