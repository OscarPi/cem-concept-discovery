import pickle
from pathlib import Path
import torch
from tqdm import tqdm
import clip
from cemcd.data import transforms
from PIL import Image

# Run foundation model on a list of images
def run_foundation_model_and_save(data_list, dataset_dir, foundation_model, out_file, model_dir="/checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if foundation_model == "dinov2_vitg14":
        torch.hub.set_dir(Path(model_dir) / "dinov2")
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(device)
        model.eval()
        transform = transforms.dino_transforms
    elif foundation_model == "clip_vitl14":
        ckpt_dir = Path(model_dir) / "clip"
        model, transform = clip.load("ViT-L/14", device=device, download_root=ckpt_dir)
        model.eval()
        model = model.encode_image
        transform.transforms[2] = transforms._convert_image_to_rgb
        transform.transforms[3] = transforms._safe_to_tensor
    else:
        raise ValueError(f"Unrecognised foundation model: {model}.")


    xs = []
    with torch.no_grad():
        for img_data in tqdm(data_list):
            image_path = dataset_dir / "imagenet" / img_data["image_path"]
            img = Image.open(image_path).convert("RGB")

            img = transform(img)

            img = img[torch.newaxis, ...].to(device)
            x = model(img).detach().cpu().squeeze().float()

            xs.append(x)

    xs = torch.stack(xs)

    torch.save(xs, out_file)

def main():
    DATASET_DIR = "/datasets"
    OUTPUT_DIR = "/datasets/imagenet"
    FOUNDATION_MODEL = "clip_vitl14"
    MODEL_DIR = "/checkpoints"
    GROUP_SIZE = 50000

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    splits_file = Path(DATASET_DIR) / "imagenet" / "splits.pkl"
    if not splits_file.exists():
        raise FileNotFoundError(f"Cannot find {splits_file}. Run the split script first.")

    with open(splits_file, "rb") as f:
        splits = pickle.load(f)

    for split_name in ["val", "test", "train"]:
        split_data = splits[split_name]
        total = len(split_data)
        n_groups = (total + GROUP_SIZE - 1) // GROUP_SIZE

        print(f"\nProcessing {split_name} split with {total} images ({n_groups} groups)...")

        for g in range(n_groups):
            out_file = Path(OUTPUT_DIR) / f"{split_name}_{FOUNDATION_MODEL}_features_group_{g+1}.pt"
            if out_file.exists():
                raise FileExistsError(f"Split {split_name} Group {g+1} already exists.")

            start = g * GROUP_SIZE
            end = min((g + 1) * GROUP_SIZE, total)
            group_data = split_data[start:end]

            run_foundation_model_and_save(
                data_list=group_data,
                dataset_dir=Path(DATASET_DIR),
                foundation_model=FOUNDATION_MODEL,
                out_file=out_file,
                model_dir=MODEL_DIR,
            )

    print("\n🎉 All splits processed and saved.")

if __name__ == "__main__":
    main()
