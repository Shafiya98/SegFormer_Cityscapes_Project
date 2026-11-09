import os
import random
import numpy as np

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transformers import SegformerForSemanticSegmentation

from tqdm import tqdm

import matplotlib.pyplot as plt

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(device)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

DATA_ROOT = "/home/veningston/Aquib_Lossfunctions/SegFormer_Cityscapes_Project/Data/cityscapes"

IMAGE_SIZE = 512

NUM_CLASSES = 19

BATCH_SIZE = 4

NUM_EPOCHS = 150

LEARNING_RATE = 1e-5

WEIGHT_DECAY = 0.01

IGNORE_INDEX = 255

image_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

mask_transform = transforms.Compose([
    transforms.Resize(
        (512, 512),
        interpolation=Image.NEAREST
    ),
    transforms.PILToTensor()
])


class CityscapesDataset(Dataset):

    def __init__(self, root, split="train",
                 image_transform=None,
                 mask_transform=None):

        self.image_dir = os.path.join(
            root, "leftImg8bit", split
        )

        self.label_dir = os.path.join(
            root, "gtFine", split
        )

        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.images = []

        for city in os.listdir(self.image_dir):

            city_path = os.path.join(
                self.image_dir, city
            )

            for img in os.listdir(city_path):

                if img.endswith("_leftImg8bit.png"):

                    self.images.append((city, img))

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        city, img_name = self.images[idx]

        img_path = os.path.join(
            self.image_dir,
            city,
            img_name
        )

        label_name = img_name.replace(
            "_leftImg8bit.png",
            "_gtFine_labelIds.png"
        )

        label_path = os.path.join(
            self.label_dir,
            city,
            label_name
        )

        image = Image.open(img_path).convert("RGB")

        mask = Image.open(label_path)

        mask = np.array(mask)

        id_map = {
            7:0,
            8:1,
            11:2,
            12:3,
            13:4,
            17:5,
            19:6,
            20:7,
            21:8,
            22:9,
            23:10,
            24:11,
            25:12,
            26:13,
            27:14,
            28:15,
            31:16,
            32:17,
            33:18
        }

        new_mask = np.ones_like(
            mask,
            dtype=np.uint8
        ) * 255

        for k, v in id_map.items():

            new_mask[mask == k] = v

        mask = Image.fromarray(new_mask)

        if self.image_transform:

            image = self.image_transform(image)

        if self.mask_transform:

            mask = self.mask_transform(mask)

        return image, mask

train_dataset = CityscapesDataset(
    root=DATA_ROOT,
    split="train",
    image_transform=image_transform,
    mask_transform=mask_transform
)

val_dataset = CityscapesDataset(
    root=DATA_ROOT,
    split="val",
    image_transform=image_transform,
    mask_transform=mask_transform
)

print("Train Images :", len(train_dataset))
print("Validation Images :", len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

images, masks = next(iter(train_loader))

print(images.shape)
print(masks.shape)
print(torch.unique(masks))

from transformers import SegformerForSemanticSegmentation

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
)

model = model.to(device)

print("Fresh Pretrained Model Loaded Successfully ")

class FocalLoss(nn.Module):

    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=255):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):

        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            ignore_index=self.ignore_index
        )

        pt = torch.exp(-ce_loss)

        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        valid = targets != self.ignore_index

        return focal_loss[valid].mean()

criterion = FocalLoss(
    alpha=1.0,
    gamma=2.0,
    ignore_index=255
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5,
    weight_decay=0.01
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=150
)

print("Focal Loss Initialized Successfully ")

import os
import numpy as np
import torch
import torch.nn.functional as F

os.makedirs("results/checkpoints", exist_ok=True)

start_epoch = 0
num_epochs = 150
num_classes = 19

best_val_miou = 0.0

train_losses_focal = []
val_pixel_acc_focal = []
val_miou_focal = []

for epoch in range(start_epoch, num_epochs):

    model.train()

    running_loss = 0.0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.squeeze(1).long().to(device)

        optimizer.zero_grad()

        outputs = model(pixel_values=images)

        logits = F.interpolate(
            outputs.logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        loss = criterion(logits, masks)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    scheduler.step()

    epoch_loss = running_loss / len(train_loader)
    train_losses_focal.append(epoch_loss)

    model.eval()

    correct = 0
    total = 0

    intersections = np.zeros(num_classes)
    unions = np.zeros(num_classes)

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.squeeze(1).long().to(device)

            outputs = model(pixel_values=images)

            logits = F.interpolate(
                outputs.logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            preds = torch.argmax(logits, dim=1)

            valid = (masks != 255)

            correct += (((preds == masks) & valid).sum().item())
            total += valid.sum().item()

            for cls in range(num_classes):

                pred_cls = (preds == cls) & valid
                true_cls = (masks == cls) & valid

                intersections[cls] += (pred_cls & true_cls).sum().item()
                unions[cls] += (pred_cls | true_cls).sum().item()

    val_pixel_acc = correct / total
    val_pixel_acc_focal.append(val_pixel_acc)

    ious = []

    for cls in range(num_classes):

        if unions[cls] > 0:
            ious.append(intersections[cls] / unions[cls])
        else:
            ious.append(0.0)

    epoch_miou = np.mean(ious)

    val_miou_focal.append(epoch_miou)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch+1:03d}/{num_epochs} | "
        f"Loss: {epoch_loss:.4f} | "
        f"Val Pixel Acc: {val_pixel_acc:.4f} | "
        f"Val mIoU: {epoch_miou:.4f} | "
        f"LR: {current_lr:.8f}"
    )

    if epoch_miou > best_val_miou:

        best_val_miou = epoch_miou

        torch.save(
            model.state_dict(),
            "results/checkpoints/best_cityscapes_focal_150.pth"
        )

        print(f"?? Best Focal Model Saved (mIoU: {best_val_miou:.4f})")

    # Har epoch ke baad latest model save hoga
    torch.save(
        model.state_dict(),
        "results/checkpoints/cityscapes_focal_latest.pth"
    )

    if (epoch + 1) % 10 == 0:

        torch.save(
            model.state_dict(),
            f"results/checkpoints/focal_epoch_{epoch+1}.pth"
        )

        print(f"?? Epoch {epoch+1} Backup Saved")

np.save(
    "results/focal_train_losses.npy",
    np.array(train_losses_focal)
)

np.save(
    "results/focal_val_pixel_acc.npy",
    np.array(val_pixel_acc_focal)
)

np.save(
    "results/focal_val_miou.npy",
    np.array(val_miou_focal)
)

print("\n==========================================")
print("?? Focal Loss Training Completed!")
print(f"Best Validation mIoU : {best_val_miou:.4f}")
print("==========================================")


import matplotlib.pyplot as plt

# Training Loss Curve
plt.figure(figsize=(8,5))

plt.plot(
    train_losses_focal,
    linewidth=2,
    label="Training Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Focal Loss Training Curve")

plt.grid(True)
plt.legend()

plt.savefig(
    "results/training_loss_curve_focal.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Training Loss Curve Saved")


# Validation mIoU Curve
plt.figure(figsize=(8,5))

plt.plot(
    val_miou_focal,
    linewidth=2,
    label="Validation mIoU"
)

plt.xlabel("Epoch")
plt.ylabel("mIoU")
plt.title("Validation mIoU Curve")

plt.grid(True)
plt.legend()

plt.savefig(
    "results/validation_iou_curve_focal.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Validation mIoU Curve Saved")

with open("results/focal_results.txt", "w") as f:

    f.write("========== FOCAL LOSS RESULTS ==========\n")
    f.write(f"Best Validation mIoU : {best_val_miou:.4f}\n")
    f.write(f"Total Epochs : {num_epochs}\n")
    f.write(f"Learning Rate : 1e-5\n")
    f.write("Optimizer : AdamW\n")
    f.write("Scheduler : CosineAnnealingLR\n")
    f.write("Loss Function : Focal Loss\n")

print("Results Saved Successfully")
