import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.data import ChestXrayDataset, CLASS_NAMES, filter_df_to_existing_images
from src.model import build_model
from src.train_utils import get_device
from src.metrics import sigmoid
from src.gradcam_utils import GradCAM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--img_root", required=True)
    parser.add_argument("--weights", default="model_best.pth")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    print("Device:", device)

    df = pd.read_csv(args.csv_path)[["Image Index", "Finding Labels"]]
    df = filter_df_to_existing_images(df, args.img_root)
    print("Rows after filtering to existing images:", len(df))

    # Match the same split logic (same seed)
    _, val_df = train_test_split(df, test_size=0.2, random_state=args.seed, shuffle=True)

    # Pick a small set to visualize
    val_df = val_df.sample(n=min(args.samples, len(val_df)), random_state=args.seed).reset_index(drop=True)

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    ds = ChestXrayDataset(val_df, img_root=args.img_root, train=False, tfm=tfm)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    model = build_model(len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # EfficientNet-B0 last conv block (good default hook point)
    target_layer = model.features[-1]
    cam = GradCAM(model, target_layer)

    os.makedirs("results/gradcam", exist_ok=True)

    for i, (img, _) in enumerate(loader):
        img = img.to(device)

        # choose the top predicted class for that image
        with torch.no_grad():
            logits = model(img).detach().cpu().numpy()[0]
            probs = sigmoid(logits)
            class_idx = int(np.argmax(probs))

        heat = cam.generate(img, class_idx)

        # overlay
        img_np = img[0].detach().cpu().permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

        plt.figure(figsize=(4, 4))
        plt.imshow(img_np)
        plt.imshow(heat, alpha=0.35)
        plt.axis("off")
        plt.title(f"Grad-CAM: {CLASS_NAMES[class_idx]} ({probs[class_idx]:.2f})")

        out_path = os.path.join("results", "gradcam", f"gradcam_{i}_{CLASS_NAMES[class_idx]}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()

        print("Saved:", out_path)


if __name__ == "__main__":
    main()
