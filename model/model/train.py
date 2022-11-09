import os
import torch
import data_setup, engine, model_builder, utils
import torchvision

from torchvision import transforms
from pathlib import Path
from timeit import default_timer as timer
from torchinfo import summary

# Setup hyperparameters
NUM_EPOCHS = 50
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Setup path to data folder
data_path = "/Users/shreyachauhan/Thesis_Document_Image_Classification/model/data"
image_path = os.path.join(data_path, "Tobacco3482_100")

# Setup train and testing paths
train_dir = os.path.join(image_path, "Train")
test_dir = os.path.join(image_path, "Test")

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

# Get a set of pretrained model weights
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
auto_transforms = weights.transforms()

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=auto_transforms,
    batch_size=BATCH_SIZE,
)

# Get the length of class_names (one output unit for each class)
output_shape = len(class_names)

model = torchvision.models.efficientnet_b0(weights=weights).to(device)

# Recreate the classifier layer and seed it to the target device
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(
        in_features=1280,
        out_features=output_shape,  # same number of output units as our number of classes
        bias=True,
    ),
).to(device)

# Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
for param in model.features.parameters():
    param.requires_grad = False


# Create model with help from model_builder.py
# model = model_builder.TinyVGG(
#     input_shape=3, hidden_units=HIDDEN_UNITS, output_shape=len(class_names)
# ).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Start the timer
start_time = timer()
# Start training with help from engine.py
engine.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=NUM_EPOCHS,
    device=device,
)
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# Save the model with help from utils.py
utils.save_model(
    model=model,
    target_dir="/Users/shreyachauhan/Thesis_Document_Image_Classification/model/models",
    model_name="efficientnet_b0.pth",
)
