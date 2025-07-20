import os
import cv2
import numpy as np
from tqdm import tqdm

# -------------------------------
# Configurations
# -------------------------------
SOURCE_FOLDER = r"E:\Projects\OCR-From-Scratch\Data\Main"
DEST_FOLDER = r"E:\Projects\OCR-From-Scratch\ProcessedDataset"
IMAGE_SIZE = (32, 32)
ROTATION_ANGLES = [0, 90, 180, 270]

os.makedirs(DEST_FOLDER, exist_ok=True)

# -------------------------------
# Preprocessing Functions
# -------------------------------

def to_grayscale(image):
    """Convert image to grayscale"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def resize_with_padding(img, size=(32, 32), pad_value=255):
    """Resize and pad image to keep aspect ratio intact"""
    h, w = img.shape[:2]
    scale = min(size[0] / h, size[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    # Add padding
    pad_top = (size[0] - nh) // 2
    pad_bottom = size[0] - nh - pad_top
    pad_left = (size[1] - nw) // 2
    pad_right = size[1] - nw - pad_left
    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=pad_value)
    
    return padded

def rotate_image(img, angle):
    """Rotate image by specific angle"""
    if angle == 0:
        return img
    rot_code = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE
    }
    return cv2.rotate(img, rot_code[angle])

# -------------------------------
# Process Single Image (Optional)
# -------------------------------

def process_and_save(image_path, dest_path_prefix):
    """Apply grayscale, resize, rotate, and save multiple versions of image"""
    img = cv2.imread(image_path)
    if img is None:
        return
    
    gray = to_grayscale(img)
    processed = resize_with_padding(gray, IMAGE_SIZE)

    for angle in ROTATION_ANGLES:
        rotated = rotate_image(processed, angle)
        angle_tag = f"_{angle}" if angle != 0 else ""
        filename = f"{dest_path_prefix}{angle_tag}.png"
        cv2.imwrite(filename, rotated)

# -------------------------------
# Main Dataset Processing
# -------------------------------

def process_dataset():
    for folder in tqdm(os.listdir(SOURCE_FOLDER), desc="Processing folders"):
        src_folder = os.path.join(SOURCE_FOLDER, folder)
        dst_folder = os.path.join(DEST_FOLDER, folder)
        os.makedirs(dst_folder, exist_ok=True)

        if not os.path.isdir(src_folder):
            continue

        for filename in os.listdir(src_folder):
            name, ext = os.path.splitext(filename)
            src_path = os.path.join(src_folder, filename)
            dst_path_prefix = os.path.join(dst_folder, name)
            process_and_save(src_path, dst_path_prefix)

# -------------------------------
# Example: Test on Random Image (Optional)
# -------------------------------

def test_single_image(image_path, output_dir="TestOutput"):
    """Process and show/save result of a single real image"""
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(image_path))[0]

    img = cv2.imread(image_path)
    if img is None:
        print("Invalid image.")
        return
    
    gray = to_grayscale(img)
    processed = resize_with_padding(gray, IMAGE_SIZE)

    for angle in ROTATION_ANGLES:
        rotated = rotate_image(processed, angle)
        angle_tag = f"_{angle}" if angle != 0 else ""
        cv2.imwrite(os.path.join(output_dir, f"{name}{angle_tag}.png"), rotated)

    print("✅ Image processed and saved to", output_dir)

# -------------------------------
# Run Everything
# -------------------------------

if __name__ == "__main__":
    process_dataset()
    # Example: test_single_image(r"path/to/any/random/image.png")
