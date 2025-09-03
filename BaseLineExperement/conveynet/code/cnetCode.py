
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# ----------------
# Model builder
# ----------------
def build_convnext_base(num_classes=5):
    # Try torchvision first
    try:
        from torchvision.models import convnext_base
        m = convnext_base(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m
    except Exception:
        # Fallback to timm
        import timm
        return timm.create_model("convnext_base", pretrained=False, num_classes=num_classes)

# ----------------
# Dataset
# ----------------
class APTOSValDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str, image_size: int = 384):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def _resolve(self, id_code: str) -> str:
        for ext in (".png",".jpg",".jpeg",".PNG",".JPG",".JPEG"):
            p = os.path.join(self.image_dir, id_code + ext)
            if os.path.isfile(p): return p
        return ""

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        p = self._resolve(str(r["id_code"]))
        if p == "":
            raise FileNotFoundError(f"Image not found for id_code={r['id_code']}")
        img = Image.open(p).convert("RGB")
        return self.tf(img), int(r["diagnosis"]), str(r["id_code"])

# ----------------
# Main
# ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", required=True, type=str)
    ap.add_argument("--image_dir", required=True, type=str)
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--output_csv", required=True, type=str)
    ap.add_argument("--batch_size", default=32, type=int)
    ap.add_argument("--num_workers", default=2, type=int)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device.type)

    df = pd.read_csv(args.val_csv)
    ds = APTOSValDataset(df, args.image_dir, image_size=384)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    model = build_convnext_base(num_classes=5).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    print("Loaded:", args.checkpoint)

    all_ids, all_true, all_pred = [], [], []
    with torch.no_grad():
        for imgs, labels, ids in tqdm(dl, desc="Infer"):
            imgs = imgs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_pred.append(preds); all_true.append(labels.numpy()); all_ids += list(ids)

    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id_code": all_ids, "true": y_true, "pred": y_pred}).to_csv(out_path, index=False)
    print("✅ Saved predictions to:", out_path)

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, digits=4))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
