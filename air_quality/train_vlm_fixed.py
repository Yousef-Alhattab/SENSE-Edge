"""
VLM-BASED CLASSIFICATION FOR BEIJING PM2.5
Using CLIP (Vision-Language Model) on time-series image encodings
This is a TRUE VLM implementation.
"""

import os
import random
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from transformers import CLIPProcessor, CLIPModel

import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


# ================= CONFIG =================
DATA_ROOT = "data/vlm_images_all_6_methods"
ENCODING_METHOD = "gaf_summation"   # change if needed

TRAIN_DIR = os.path.join(DATA_ROOT, ENCODING_METHOD, "train")
TEST_DIR  = os.path.join(DATA_ROOT, ENCODING_METHOD, "test")

MODEL_NAME = "openai/clip-vit-base-patch32"

BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-5
WEIGHT_DECAY = 0.01
PATIENCE = 20

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASELINE_ACC = 48.8
# ==========================================


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ✅ IMPORTANT FIX: CUSTOM COLLATE FUNCTION (FOR PIL IMAGES)
def clip_collate_fn(batch):
    images = [item[0] for item in batch]   # list of PIL images
    labels = torch.tensor([item[1] for item in batch])
    return images, labels


class StrongAugmentationForCLIP:
    def __init__(self, img_size=224):
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.1),
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
        ])

    def get_train_transform(self):
        return self.train_transform

    def get_test_transform(self):
        return self.test_transform


def build_prompts(class_names):
    prompts = []
    for cname in class_names:
        prompts.append(
            f"an air pollution level {cname} in Beijing represented as a time series image"
        )
    return prompts


def compute_text_features(model, processor, prompts, device):
    text_inputs = processor(
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    with torch.no_grad():
        text_features = model.get_text_features(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"]
        )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features


def train_one_epoch(model, processor, text_features, loader, optimizer, device, epoch):

    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(loader, desc=f"Epoch {epoch}")

    for images, labels in pbar:
        labels = labels.to(device)

        image_inputs = processor(
            images=images,
            return_tensors="pt"
        ).to(device)

        image_features = model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100 * correct / total:.2f}%"
        })

    return total_loss / total, 100 * correct / total


def evaluate(model, processor, text_features, loader, device):

    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    all_preds = []
    all_targets = []

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            labels = labels.to(device)

            image_inputs = processor(
                images=images,
                return_tensors="pt"
            ).to(device)

            image_features = model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logit_scale = model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)

            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    return total_loss / total, 100 * correct / total, all_preds, all_targets


def main():
    set_seed(SEED)

    print("=" * 80)
    print("🔥 VLM (CLIP) CLASSIFICATION - BEIJING PM2.5")
    print("=" * 80)

    aug = StrongAugmentationForCLIP(224)

    train_dataset = ImageFolder(TRAIN_DIR, transform=aug.get_train_transform())
    test_dataset = ImageFolder(TEST_DIR, transform=aug.get_test_transform())

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print("Classes:", class_names)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        collate_fn=clip_collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=clip_collate_fn,
        pin_memory=True
    )

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)

    for p in model.text_model.parameters():
        p.requires_grad = False

    prompts = build_prompts(class_names)
    text_features = compute_text_features(model, processor, prompts, DEVICE)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_acc = 0
    patience = 0

    for epoch in range(1, EPOCHS + 1):

        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, processor, text_features, train_loader, optimizer, DEVICE, epoch
        )

        val_loss, val_acc, preds, targets = evaluate(
            model, processor, text_features, test_loader, DEVICE
        )

        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0

            torch.save(model.state_dict(), f"best_clip_{ENCODING_METHOD}.pth")
            print("✅ New Best Model Saved")

        else:
            patience += 1

        if patience >= PATIENCE:
            print("⛔ Early stopping")
            break

    print("\n====== FINAL REPORT ======")
    print(f"Best Accuracy: {best_acc:.2f}%")
    print(f"Baseline: {BASELINE_ACC}%")
    print(f"Improvement: {best_acc - BASELINE_ACC:+.2f}%")

    print("\nClassification Report:")
    print(classification_report(targets, preds, target_names=class_names))

    print("\nConfusion Matrix:")
    print(confusion_matrix(targets, preds))


if __name__ == "__main__":
    main()
