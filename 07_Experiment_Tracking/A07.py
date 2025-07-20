import os
import zipfile
from pathlib import Path

import requests
import torch
import torchvision
from torch import nn

import wandb
from Going_Modular import data_setup
from Going_Modular.engine import train
from Going_Modular.helper_functions import set_seeds

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"

def download_data(source: str,
                  destination: str,
                  remove_source: bool = True) -> Path:
    """Downloads a zipped dataset from source and unzips to destination."""
    data_path = Path("data/")
    image_path = data_path / destination

    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file} from {source}...")
            f.write(request.content)
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file} data...")
            zip_ref.extractall(image_path)
        if remove_source:
            os.remove(data_path / target_file)
    return image_path

def main():
    # Step 0. Download and extract data once (before looping experiments)
    image_path = download_data(
        source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
        destination="pizza_steak_sushi")

    # Step 1. Setup training/testing directories
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    # Step 2. Series of experiment configurations
    experiment_configs = [
        {"architecture": "EfficientNet_B0", "epochs": 5, "learning_rate": 0.001, "batch_size": 32},
        {"architecture": "EfficientNet_B0", "epochs": 5, "learning_rate": 0.001, "batch_size": 16},
        {"architecture": "EfficientNet_B0", "epochs": 5, "learning_rate": 0.01, "batch_size": 32},
        {"architecture": "EfficientNet_B0", "epochs": 5, "learning_rate": 0.01, "batch_size": 16},
        {"architecture": "EfficientNet_B0", "epochs": 10, "learning_rate": 0.001, "batch_size": 32},
        {"architecture": "EfficientNet_B0", "epochs": 10, "learning_rate": 0.001, "batch_size": 16},
        {"architecture": "EfficientNet_B0", "epochs": 10, "learning_rate": 0.01, "batch_size": 32},
        {"architecture": "EfficientNet_B0", "epochs": 10, "learning_rate": 0.01, "batch_size": 16},
    ]

    for config in experiment_configs:
        # Step 3. Initialize wandb at the start of each run
        wandb.init(project="food-vision", config=config,
                   name=f"{config['architecture']}"
                        f"_ep{config['epochs']}"
                        f"_lr{config['learning_rate']}"
                        f"_bs{config['batch_size']}")

        wandb.define_metric("train_loss", step_metric="epoch")
        wandb.define_metric("train_acc", step_metric="epoch")
        wandb.define_metric("test_loss", step_metric="epoch")
        wandb.define_metric("test_acc", step_metric="epoch")

        # Step 4. Prepare model weights and transforms
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        automatic_transforms = weights.transforms()

        # Step 5. Create dataloaders
        train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
            train_dir=train_dir,
            test_dir=test_dir,
            transform=automatic_transforms,
            batch_size=wandb.config.batch_size,
            num_workers=0 #os.cpu_count(), but only for BIG datasets
        )

        # Step 6. Initialize and customize model
        model = torchvision.models.efficientnet_b0(weights=weights).to(device)
        for param in model.features.parameters():
            param.requires_grad = False
        set_seeds()
        model.classifier = torch.nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=len(class_names), bias=True).to(device))

        # Step 7. Log model structure to wandb
        wandb.watch(model, log="all", log_freq=10)

        # Step 8. Define loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

        # Step 9. Train model
        set_seeds()
        results = train(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=wandb.config.epochs,
            device=device,
            # If your 'train' function supports callbacks or logger, pass wandb logger here
        )

        # Step 10. Log final training results (epoch-wise metrics, with epoch as X axis)
        max_epochs = max(len(v) for v in results.values() if isinstance(v, (list, tuple)))
        for epoch in range(max_epochs):
            log_dict = {
                "epoch": epoch,
                "train_loss": results["train_loss"][epoch] if "train_loss" in results and epoch < len(
                    results["train_loss"]) else None,
                "train_acc": results["train_acc"][epoch] if "train_acc" in results and epoch < len(
                    results["train_acc"]) else None,
                "test_loss": results["test_loss"][epoch] if "test_loss" in results and epoch < len(
                    results["test_loss"]) else None,
                "test_acc": results["test_acc"][epoch] if "test_acc" in results and epoch < len(
                    results["test_acc"]) else None,
            }
            wandb.log(log_dict)

        # Step 11. Save the trained model and log as artifact/checkpoint
        model_path = f"./models/food-vision_{wandb.run.name}.pth"
        torch.save(model.state_dict(), model_path)
        wandb.save(model_path)

        # Step 12. Finish wandb run (best practice)
        wandb.finish()

if __name__ == "__main__":
    main()
