import os
import random
from PIL import Image
import torch
from PictSure import PictSure

# CONFIG
ROOT_DIR = "./BrainTumor_preprocessed/"
NUM_CONTEXT_IMAGES = 5
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Load context/reference images
def load_reference_images(path):
    label_map = {}
    context_images, context_labels = [], []
    
    folders = sorted(os.listdir(path))
    for label, folder in enumerate(folders):
        folder_path = os.path.join(path, folder)
        all_images = os.listdir(folder_path)
        chosen = random.sample(all_images, NUM_CONTEXT_IMAGES + 1)  # +1 for extra test image
        ref_imgs = chosen[:-1]
        test_img = chosen[-1]

        for img_name in ref_imgs:
            img_path = os.path.join(folder_path, img_name)
            img = Image.open(img_path).convert("RGB")
            context_images.append(img)
            context_labels.append(label)

        label_map[folder] = label

    return context_images, context_labels, label_map, chosen

# Pick a single test image (one left out per class)
def pick_test_image(path, label_map, chosen):
    all_images = []
    all_labels = []
    
    for folder, label in label_map.items():
        folder_path = os.path.join(path, folder)
        images = [f for f in os.listdir(folder_path) if f not in chosen]
        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            all_images.append(img_path)
            all_labels.append(label)
    
    if all_images:
        random_index = random.randint(0, len(all_images) - 1)
        img_path = all_images[random_index]
        label = all_labels[random_index]
        img = Image.open(img_path).convert("RGB")
        return img, label

# or pull our pre-trained models from HuggingFace
pictsure_model = PictSure.from_pretrained("pictsure/pictsure-vit").to(DEVICE)

results = []
for i in range(200):
    # Load references and test image
    context_imgs, context_lbls, label_map, chosen = load_reference_images(ROOT_DIR)
    test_img, test_lbl = pick_test_image(ROOT_DIR, label_map, chosen)
    # Predict
    with torch.no_grad():
        pictsure_model.set_context_images(context_imgs, context_lbls)
        pred = pictsure_model.predict(test_img)

    results.append((pred == test_lbl))

accuracy = sum(results) / len(results) * 100
print(f"Accuracy over {len(results)} predictions: {accuracy:.1f}%")