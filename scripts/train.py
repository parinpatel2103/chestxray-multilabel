import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data import ChestXrayDataset, CLASS_NAMES, list_image_folders, filter_df_to_existing_images
from src.model import build_model
from src.train_utils import set_seed, get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--img_root", required=True, help="Folder containing images_001, images_002, ...")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    # Subset for faster training during development
    parser.add_argument("--train_subset", type=int, default=2000, help="0 = full train split")
    parser.add_argument("--val_subset", type=int, default=500, help="0 = full val split")

    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print("Device:", device)

    # Load metadata
    df = pd.read_csv(args.csv_path)[["Image Index", "Finding Labels"]]

    # Check that folders exist
    img_dirs = list_image_folders(args.img_root)
    if len(img_dirs) == 0:
        raise RuntimeError(
            "No image folders found. Expected something like:\n"
            "<img_root>/images_001/images, <img_root>/images_002/images, ..."
        )

    # Filter rows to images that actually exist on disk
    df = filter_df_to_existing_images(df, args.img_root)
    print("Rows after filtering to existing images:", len(df))

    # Split
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=args.seed, shuffle=True
    )

    # Fast subset (default ON). Set to 0 to use full split.
    if args.train_subset and args.train_subset > 0:
        train_df = train_df.sample(
            n=min(args.train_subset, len(train_df)), random_state=0
        ).reset_index(drop=True)

    if args.val_subset and args.val_subset > 0:
        val_df = val_df.sample(
            n=min(args.val_subset, len(val_df)), random_state=0
        ).reset_index(drop=True)

    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")

    # Minimal transforms 
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_ds = ChestXrayDataset(train_df, img_root=args.img_root, train=True, tfm=tfm)
    val_ds   = ChestXrayDataset(val_df,   img_root=args.img_root, train=False, tfm=tfm)

    pin = (device.type == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    model = build_model(len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        running = 0.0

        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            imgs, targets = imgs.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running += loss.item()

        avg_train_loss = running / max(len(train_loader), 1)

        # Validation loss
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                logits = model(imgs)
                val_running += criterion(logits, targets).item()

        avg_val_loss = val_running / max(len(val_loader), 1)

        print(f"Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "model_best.pth")
            print("Saved model_best.pth")

    torch.save(model.state_dict(), "model_last.pth")
    print("Saved model_last.pth")


if __name__ == "__main__":
    main()
