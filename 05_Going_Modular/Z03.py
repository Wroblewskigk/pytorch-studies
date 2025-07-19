"""
Exercise 3: Create a script to predict (such as predict.py) on a target image given a file
path with a saved model.

- For example, you should be able to run the command python predict.py some_image.jpeg and have
a trained PyTorch model predict on the image and return its prediction.
- To see example prediction code, check out the predicting on a custom image section in
notebook 04.
- You may also have to write code to load in a trained model.
"""

import argparse
import torch
from torchvision import transforms
from PIL import Image
from Going_Modular import model_builder

# 1. Parse command-line arguments
parser = argparse.ArgumentParser(description="Predict the class of an input image using a "
                                             "trained PyTorch model.")

parser.add_argument("image_path",
                    type=str,
                    help="Path to the image file")

parser.add_argument("--model_path",
                    type=str,
                    default="models/05_going_modular_script_mode_tinyvgg_model.pth",
                    help="Path to the trained model file")

parser.add_argument("--hidden_units",
                    type=int,
                    default=10,
                    help="Number of hidden units in the TinyVGG model")

parser.add_argument("--class_names",
                    nargs="+",
                    default=["pizza", "steak", "sushi"],
                    help="List of class names")

args = parser.parse_args()

# 2. Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 3. Load the trained model
model = model_builder.TinyVGG(input_shape=3,
                              hidden_units=args.hidden_units,
                              output_shape=len(args.class_names))
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.to(device)
model.eval()

# 4. Define image transforms (should match training)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# 5. Load and preprocess the image
image = Image.open(args.image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# 6. Make prediction
with torch.inference_mode():
    outputs = model(image)
    predicted_idx = torch.argmax(outputs, dim=1).item()
    predicted_class = args.class_names[predicted_idx]

print(f"Predicted class index: {predicted_idx}")
print(f"Predicted class name: {predicted_class}")