

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# ------------ Dataset ------------
class APTOSValDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_dir: str, image_size: int = 384):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])

    def resolve_path(self, id_code: str) -> str:
        for ext in [".png",".jpg",".jpeg",".PNG",".JPG",".JPEG"]:
            p = os.path.join(self.image_dir, id_code + ext)
            if os.path.isfile(p):
                return p
        return ""

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        img_path = self.resolve_path(str(r["id_code"]))
        if img_path == "":
            raise FileNotFoundError(f"Image not found for {r['id_code']}")
        image = Image.open(img_path).convert("RGB")
        return self.tf(image), int(r["diagnosis"]), str(r["id_code"])

# ------------ Model ------------
def build_resnet18(num_classes=5):
    model = models.resnet18(weights=None)  # weights=None for pure inference structure
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

# ------------ Main ------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_csv", required=True, type=str)
    parser.add_argument("--image_dir", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output_csv", required=True, type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device.type)

    val_df = pd.read_csv(args.val_csv)
    val_dataset = APTOSValDataset(val_df, args.image_dir, image_size=384)
    val_loader  = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = build_resnet18(num_classes=5).to(device).eval()

    # Load checkpoint (weights-only safe)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    print("Loaded checkpoint:", args.checkpoint)

    all_preds, all_true, all_ids = [], [], []
    with torch.no_grad():
        for images, labels, ids in tqdm(val_loader, desc="Infer"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_true.append(labels.numpy())
            all_ids.extend(ids)

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    # Save CSV
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id_code": all_ids, "true": y_true, "pred": y_pred}).to_csv(output_path, index=False)
    print("✅ Predictions saved to:", output_path)

    # Print quick report
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, digits=4))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
