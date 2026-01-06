import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms

from src.data import ChestXrayDataset, CLASS_NAMES, filter_df_to_existing_images
from src.model import build_model
from src.train_utils import get_device
from src.metrics import sigmoid, tune_thresholds, compute_auc_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--img_root", required=True)
    parser.add_argument("--weights", default="model_best.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--subset", type=int, default=0, help="Eval on N samples for speed (0 = full val split)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    print("Device:", device)

    # Load & filter to existing images (prevents FileNotFound)
    df = pd.read_csv(args.csv_path)[["Image Index", "Finding Labels"]]
    df = filter_df_to_existing_images(df, args.img_root)
    print("Rows after filtering to existing images:", len(df))

    # Recreate split with same seed used in training
    _, val_df = train_test_split(df, test_size=0.2, random_state=args.seed, shuffle=True)

    if args.subset and args.subset > 0:
        val_df = val_df.sample(n=min(args.subset, len(val_df)), random_state=0).reset_index(drop=True)

    print("Val samples:", len(val_df))

    # Use same minimal transforms as training
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    val_ds = ChestXrayDataset(val_df, img_root=args.img_root, train=False, tfm=tfm)

    pin = (device.type == "cuda")
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    model = build_model(len(CLASS_NAMES)).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs).detach().cpu().numpy()
            all_logits.append(logits)
            all_labels.append(targets.numpy())

    all_logits = np.vstack(all_logits)
    all_labels = np.vstack(all_labels)

    probs = sigmoid(all_logits)

    thresholds, _ = tune_thresholds(probs, all_labels, t_min=0.05, t_max=0.40, steps=8)
    rows, macro_auc, macro_f1 = compute_auc_f1(probs, all_labels, thresholds, CLASS_NAMES)

    os.makedirs("results/tables", exist_ok=True)
    out_path = os.path.join("results", "tables", "metrics.txt")

    with open(out_path, "w") as f:
        f.write("Per-class results (AUC, F1, threshold)\n")
        f.write("-----------------------------------\n")
        for name, auc, f1, thr in rows:
            auc_str = "nan" if np.isnan(auc) else f"{auc:.3f}"
            f.write(f"{name:18s} AUC={auc_str}  F1={f1:.3f}  thr={thr:.2f}\n")

        f.write("\n")
        f.write(f"Macro AUC: {macro_auc:.3f}\n")
        f.write(f"Macro F1 : {macro_f1:.3f}\n")

    print(f"Saved metrics to {out_path}")
    print(f"Macro AUC: {macro_auc:.3f}")
    print(f"Macro F1 : {macro_f1:.3f}")


if __name__ == "__main__":
    main()
