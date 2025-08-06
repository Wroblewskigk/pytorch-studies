import os
import pathlib
import random
import shutil
from pathlib import Path
from timeit import default_timer as timer
from typing import List, Dict, Tuple

import pandas as pd
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from tqdm.auto import tqdm

from Going_Modular import data_setup, engine, utils

device = "cuda"


def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken."""
    # Start the timer
    start_time = timer()

    # Transform the target image and add a batch dimension
    img = effnetb2_transforms(img).unsqueeze(0)

    # Put model into evaluation mode and turn on inference mode
    effnetb2.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(effnetb2(img), dim=1)

    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {
        class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))
    }

    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)

    # Return the prediction dictionary and prediction time
    return pred_labels_and_probs, pred_time


def pred_and_store(
    paths: List[pathlib.Path],
    model: torch.nn.Module,
    transform: torchvision.transforms,
    class_names: List[str],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> List[Dict]:
    """A function to return a list of dictionaries with sample, truth label, prediction, prediction probability and
    prediction time"""

    # 2. Create an empty list to store prediction dictionaries
    pred_list = []

    # 3. Loop through target paths
    for path in tqdm(paths):
        # 4. Create empty dictionary to store prediction information for each sample
        pred_dict = {}

        # 5. Get the sample path and ground truth class name
        pred_dict["image_path"] = path
        class_name = path.parent.stem
        pred_dict["class_name"] = class_name

        # 6. Start the prediction timer
        start_time = timer()

        # 7. Open image path
        img = Image.open(path)

        # 8. Transform the image, add batch dimension and put image on target device
        transformed_image = transform(img).unsqueeze(0).to(device)

        # 9. Prepare model for inference by sending it to target device and turning on eval() mode
        model.to(device)
        model.eval()

        # 10. Get prediction probability, predicition label and prediction class
        with torch.inference_mode():
            pred_logit = model(transformed_image)  # perform inference on target sample
            pred_prob = torch.softmax(
                pred_logit, dim=1
            )  # turn logits into prediction probabilities
            pred_label = torch.argmax(
                pred_prob, dim=1
            )  # turn prediction probabilities into prediction label
            pred_class = class_names[
                pred_label.cpu()
            ]  # hardcode prediction class to be on CPU

            # 11. Make sure things in the dictionary are on CPU (required for inspecting predictions later on)
            pred_dict["pred_prob"] = round(pred_prob.unsqueeze(0).max().cpu().item(), 4)
            pred_dict["pred_class"] = pred_class

            # 12. End the timer and calculate time per pred
            end_time = timer()
            pred_dict["time_for_pred"] = round(end_time - start_time, 4)

        # 13. Does the pred match the true label?
        pred_dict["correct"] = class_name == pred_class

        # 14. Add the dictionary to the list of preds
        pred_list.append(pred_dict)

    # 15. Return list of prediction dictionaries
    return pred_list


def create_effnetb2_model(num_classes: int = 3, seed: int = 42):
    """Creates an EfficientNetB2 feature extractor model and transforms.

    Args:
        num_classes (int, optional): number of classes in the classifier head.
            Defaults to 3.
        seed (int, optional): random seed value. Defaults to 42.

    Returns:
        model (torch.nn.Module): EffNetB2 feature extractor model.
        transforms (torchvision.transforms): EffNetB2 image transforms.
    """
    # 1, 2, 3. Create EffNetB2 pretrained weights, transforms and model
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.efficientnet_b2(weights=weights)

    # 4. Freeze all layers in base model
    for param in model.parameters():
        param.requires_grad = False

    # 5. Change classifier head with random seed for reproducibility
    torch.manual_seed(seed)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=num_classes),
    )

    return model, transforms


def create_vit_model(num_classes: int = 3, seed: int = 42):
    """Creates a ViT-B/16 feature extractor model and transforms.

    Args:
        num_classes (int, optional): number of target classes. Defaults to 3.
        seed (int, optional): random seed value for output layer. Defaults to 42.

    Returns:
        model (torch.nn.Module): ViT-B/16 feature extractor model.
        transforms (torchvision.transforms): ViT-B/16 image transforms.
    """
    # Create ViT_B_16 pretrained weights, transforms and model
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.vit_b_16(weights=weights)

    # Freeze all layers in model
    for param in model.parameters():
        param.requires_grad = False

    # Change classifier head to suit our needs (this will be trainable)
    torch.manual_seed(seed)
    model.heads = nn.Sequential(
        nn.Linear(
            in_features=768,  # keep this the same as original model
            out_features=num_classes,
        )
    )  # update to reflect target number of classes

    return model, transforms


if __name__ == "__main__":

    # Download pizza, steak, sushi images from GitHub
    from Going_Modular.helper_functions import (
        download_data,
        set_seeds,
        plot_loss_curves,
    )

    data_20_percent_path = download_data(
        source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip",
        destination="pizza_steak_sushi_20_percent",
    )

    # Setup directory paths to train and test images
    train_dir = data_20_percent_path / "train"
    test_dir = data_20_percent_path / "test"

    """
    EffNetB2 Training===================================================================================================
    """

    effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=3, seed=42)

    # Setup DataLoaders
    train_dataloader_effnetb2, test_dataloader_effnetb2, class_names = (
        data_setup.create_dataloaders(
            train_dir=train_dir,
            test_dir=test_dir,
            transform=effnetb2_transforms,
            batch_size=32,
        )
    )

    # Setup optimizer
    optimizer = torch.optim.Adam(params=effnetb2.parameters(), lr=1e-3)
    # Setup loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Set seeds for reproducibility and train the model
    set_seeds()
    effnetb2_results = engine.train(
        model=effnetb2,
        train_dataloader=train_dataloader_effnetb2,
        test_dataloader=test_dataloader_effnetb2,
        epochs=10,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
    )

    plot_loss_curves(effnetb2_results)

    # Save the model
    utils.save_model(
        model=effnetb2,
        target_dir="models",
        model_name="09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth",
    )

    # Get the model size in bytes then convert to megabytes
    pretrained_effnetb2_model_size = Path(
        "models/09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth"
    ).stat().st_size // (
        1024 * 1024
    )  # division converts bytes to megabytes (roughly)
    print(
        f"Pretrained EffNetB2 feature extractor model size: {pretrained_effnetb2_model_size} MB"
    )

    # Count number of parameters in EffNetB2
    effnetb2_total_params = sum(torch.numel(param) for param in effnetb2.parameters())

    # Create a dictionary with EffNetB2 statistics
    effnetb2_stats = {
        "test_loss": effnetb2_results["test_loss"][-1],
        "test_acc": effnetb2_results["test_acc"][-1],
        "number_of_parameters": effnetb2_total_params,
        "model_size (MB)": pretrained_effnetb2_model_size,
    }

    """
    ViT Training========================================================================================================
    """

    # Create ViT model and transforms
    vit, vit_transforms = create_vit_model(num_classes=3, seed=42)

    # Setup ViT DataLoaders
    train_dataloader_vit, test_dataloader_vit, class_names = (
        data_setup.create_dataloaders(
            train_dir=train_dir,
            test_dir=test_dir,
            transform=vit_transforms,
            batch_size=32,
        )
    )

    # Setup optimizer
    optimizer = torch.optim.Adam(params=vit.parameters(), lr=1e-3)
    # Setup loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train ViT model with seeds set for reproducibility
    set_seeds()
    vit_results = engine.train(
        model=vit,
        train_dataloader=train_dataloader_vit,
        test_dataloader=test_dataloader_vit,
        epochs=10,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
    )

    plot_loss_curves(vit_results)

    # Save the model
    utils.save_model(
        model=vit,
        target_dir="models",
        model_name="09_pretrained_vit_feature_extractor_pizza_steak_sushi_20_percent.pth",
    )

    # Get the model size in bytes then convert to megabytes
    pretrained_vit_model_size = Path(
        "models/09_pretrained_vit_feature_extractor_pizza_steak_sushi_20_percent.pth"
    ).stat().st_size // (1024 * 1024)
    print(
        f"Pretrained ViT feature extractor model size: {pretrained_vit_model_size} MB"
    )

    # Count number of parameters in ViT
    vit_total_params = sum(torch.numel(param) for param in vit.parameters())

    # Create ViT statistics dictionary
    vit_stats = {
        "test_loss": vit_results["test_loss"][-1],
        "test_acc": vit_results["test_acc"][-1],
        "number_of_parameters": vit_total_params,
        "model_size (MB)": pretrained_vit_model_size,
    }

    """
    Models Inference====================================================================================================
    """

    # Get all test data paths
    print(f"[INFO] Finding all filepaths ending with '.jpg' in directory: {test_dir}")
    test_data_paths = list(Path(test_dir).glob("*/*.jpg"))

    # Make predictions across test dataset with EffNetB2
    effnetb2_test_pred_dicts = pred_and_store(
        paths=test_data_paths,
        model=effnetb2,
        transform=effnetb2_transforms,
        class_names=class_names,
        device="cpu",
    )

    # Turn the test_pred_dicts into a DataFrame
    effnetb2_test_pred_df = pd.DataFrame(effnetb2_test_pred_dicts)
    effnetb2_test_pred_df.head()

    # Check number of correct predictions
    effnetb2_test_pred_df.correct.value_counts()

    # Find the average time per prediction
    effnetb2_average_time_per_pred = round(
        effnetb2_test_pred_df.time_for_pred.mean(), 4
    )
    print(
        f"EffNetB2 average time per prediction: {effnetb2_average_time_per_pred} seconds"
    )

    # Add EffNetB2 average prediction time to stats dictionary
    effnetb2_stats["time_per_pred_cpu"] = effnetb2_average_time_per_pred

    # Make list of prediction dictionaries with ViT feature extractor model on test images
    vit_test_pred_dicts = pred_and_store(
        paths=test_data_paths,
        model=vit,
        transform=vit_transforms,
        class_names=class_names,
        device="cpu",
    )

    # Turn vit_test_pred_dicts into a DataFrame
    vit_test_pred_df = pd.DataFrame(vit_test_pred_dicts)
    vit_test_pred_df.head()

    # Count the number of correct predictions
    vit_test_pred_df.correct.value_counts()

    # Calculate average time per prediction for ViT model
    vit_average_time_per_pred = round(vit_test_pred_df.time_for_pred.mean(), 4)
    print(f"ViT average time per prediction: {vit_average_time_per_pred} seconds")

    # Add average prediction time for ViT model on CPU
    vit_stats["time_per_pred_cpu"] = vit_average_time_per_pred

    # Turn stat dictionaries into DataFrame
    df = pd.DataFrame([effnetb2_stats, vit_stats])

    # Add column for model names
    df["model"] = ["EffNetB2", "ViT"]

    # Convert accuracy to percentages
    df["test_acc"] = round(df["test_acc"] * 100, 2)

    # Compare ViT to EffNetB2 across different characteristics
    pd.DataFrame(
        data=(df.set_index("model").loc["ViT"] / df.set_index("model").loc["EffNetB2"]),
        # divide ViT statistics by EffNetB2 statistics
        columns=["ViT to EffNetB2 ratios"],
    ).T

    # 1. Create a plot from model comparison DataFrame
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        data=df,
        x="time_per_pred_cpu",
        y="test_acc",
        c=["blue", "orange"],  # what colours to use?
        s="model_size (MB)",
    )  # size the dots by the model sizes

    # 2. Add titles, labels and customize fontsize for aesthetics
    ax.set_title("FoodVision Mini Inference Speed vs Performance", fontsize=18)
    ax.set_xlabel("Prediction time per image (seconds)", fontsize=14)
    ax.set_ylabel("Test accuracy (%)", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True)

    # 3. Annotate with model names
    for index, row in df.iterrows():
        ax.annotate(
            text=row["model"],
            # note: depending on your version of Matplotlib, you may need to use "s=..." or "text=...",
            # see: https://github.com/faustomorales/keras-ocr/issues/183#issuecomment-977733270
            xy=(row["time_per_pred_cpu"] + 0.0006, row["test_acc"] + 0.03),
            size=12,
        )

    # 4. Create a legend based on model sizes
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.5)
    model_size_legend = ax.legend(
        handles, labels, loc="lower right", title="Model size (MB)", fontsize=12
    )

    plt.savefig("images/09-foodvision-mini-inference-speed-vs-performance.jpg")

    # Show the figure
    plt.show()

    try:
        import gradio as gr
    except ImportError:
        import subprocess

        subprocess.check_call(["python3", "-m", "pip", "install", "gradio"])
        import gradio as gr

    print(f"Gradio version: {gr.__version__}")

    # Put EffNetB2 on CPU
    effnetb2.to("cpu")

    # Get a list of all test image filepaths
    test_data_paths = list(Path(test_dir).glob("*/*.jpg"))

    # Randomly select a test image path
    random_image_path = random.sample(test_data_paths, k=1)[0]

    # Open the target image
    image = Image.open(random_image_path)
    print(f"[INFO] Predicting on image at path: {random_image_path}\n")

    # Predict on the target image and print out the outputs
    pred_dict, pred_time = predict(img=image)
    print(f"Prediction label and probability dictionary: \n{pred_dict}")
    print(f"Prediction time: {pred_time} seconds")

    # Create a list of example inputs to our Gradio demo
    example_list = [[str(filepath)] for filepath in random.sample(test_data_paths, k=3)]

    # Create title, description and article strings
    title = "FoodVision Mini 🍕🥩🍣"
    description = "An EfficientNetB2 feature extractor computer vision model to classify images of food as pizza, steak or sushi."
    article = "Created at [09. PyTorch Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/)."

    # Create the Gradio demo
    demo = gr.Interface(
        fn=predict,  # mapping function from input to output
        inputs=gr.Image(type="pil"),  # what are the inputs?
        outputs=[
            gr.Label(num_top_classes=3, label="Predictions"),  # what are the outputs?
            gr.Number(label="Prediction time (s)"),
        ],
        # our fn has two outputs, therefore we have two outputs
        examples=example_list,
        title=title,
        description=description,
        article=article,
    )

    # Launch the demo!
    demo.launch(
        debug=False, share=True  # print errors locally?
    )  # generate a publically shareable URL?

    # Create FoodVision mini demo path
    foodvision_mini_demo_path = Path("demos/foodvision_mini/")

    # Remove files that might already exist there and create new directory
    if foodvision_mini_demo_path.exists():
        shutil.rmtree(foodvision_mini_demo_path)

    # Create the directory
    foodvision_mini_demo_path.mkdir(parents=True, exist_ok=True)

    # Check what's in the folder (should be empty)
    print(os.listdir("demos/foodvision_mini/"))

    # 1. Create an examples directory
    foodvision_mini_examples_path = foodvision_mini_demo_path / "examples"
    foodvision_mini_examples_path.mkdir(parents=True, exist_ok=True)

    # 2. Collect three random test dataset image paths
    foodvision_mini_examples = [
        Path("data/pizza_steak_sushi_20_percent/test/sushi/592799.jpg"),
        Path("data/pizza_steak_sushi_20_percent/test/steak/3622237.jpg"),
        Path("data/pizza_steak_sushi_20_percent/test/pizza/2582289.jpg"),
    ]

    # 3. Copy the three random images to the examples directory
    for example in foodvision_mini_examples:
        destination = foodvision_mini_examples_path / example.name
        print(f"[INFO] Copying {example} to {destination}")
        shutil.copy2(src=example, dst=destination)

    # Get example filepaths in a list of lists
    example_list = [
        ["examples/" + example] for example in os.listdir(foodvision_mini_examples_path)
    ]

    # Create a source path for our target model
    effnetb2_foodvision_mini_model_path = "models/09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth"

    # Create a destination path for our target model
    effnetb2_foodvision_mini_model_destination = (
        foodvision_mini_demo_path / effnetb2_foodvision_mini_model_path.split("/")[1]
    )

    # Try to move the file
    try:
        print(
            f"[INFO] Attempting to move {effnetb2_foodvision_mini_model_path} to {effnetb2_foodvision_mini_model_destination}"
        )

        # Move the model
        shutil.move(
            src=effnetb2_foodvision_mini_model_path,
            dst=effnetb2_foodvision_mini_model_destination,
        )

        print(f"[INFO] Model move complete.")

    # If the model has already been moved, check if it exists
    except:
        print(
            f"[INFO] No model found at {effnetb2_foodvision_mini_model_path}, perhaps its already been moved?"
        )
        print(
            f"[INFO] Model exists at {effnetb2_foodvision_mini_model_destination}: {effnetb2_foodvision_mini_model_destination.exists()}"
        )
