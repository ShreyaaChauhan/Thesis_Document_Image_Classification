from __future__ import annotations

import os

import data_setup
import torch
import torchvision
from helper_functions import make_balanced_sampler
from helper_functions import print_class_image_count
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
# from pathlib import Path
# from timeit import default_timer as timer

# import engine
# import model_builder
# import utils
# from torch.utils.data import Dataset
# from torch.utils.data import random_split
# from torch.utils.data import SubsetRandomSampler
# from torch.utils.data import WeightedRandomSampler
# from torchinfo import summary

# Setup hyperparameters
NUM_EPOCHS = 50
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Setup path to data folder
data_path = (
    '/Users/shreyachauhan/Thesis_Document_Image_Classification/model/data'  # noqa
)
image_path = os.path.join(data_path, 'Tobacco3482_full')

# Setup train and testing paths
train_dir = os.path.join(image_path, 'Train')
test_dir = os.path.join(image_path, 'Test')

# Setup target device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    ],
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

# data = {
#     "0": 0,
#     "1": 0,
#     "2": 0,
#     "3": 0,
#     "4": 0,
#     "5": 0,
#     "6": 0,
#     "7": 0,
#     "8": 0,
#     "9": 0,
# }


# def print_class_image_count(dataLoader):
#     for t in iter(dataLoader):
#         classes, counts = t[1].unique(return_counts=True)
#         for c, cs in zip(classes, counts):
#             c = c.cpu().detach().numpy()
#             cs = cs.cpu().detach().numpy()
#             data[str(c)] = int(data[str(c)]) + int(cs)
#     print(data)


# print_class_image_count(train_dataloader)

train_data = datasets.ImageFolder(
    train_dir,
    transform=auto_transforms,
    target_transform=None,
)
test_data = datasets.ImageFolder(test_dir, transform=auto_transforms)
class_names = train_data.classes
y = [data[1] for data in train_data]


sampler = make_balanced_sampler(y)
train_dataloadersss = DataLoader(
    dataset=train_data,
    batch_size=16,
    sampler=sampler,
)
train_dataloadersss.sampler.generator.manual_seed(42)
print_class_image_count(train_dataloadersss)

"""
# print_class_image_count(train_dataloadersss)
data = {
    "0": 0,
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0,
    "5": 0,
    "6": 0,
    "7": 0,
    "8": 0,
    "9": 0,
}
for t in iter(train_dataloadersss):
    classes, counts = t[1].unique(return_counts=True)
    for c, cs in zip(classes, counts):
        c = c.cpu().detach().numpy()
        cs = cs.cpu().detach().numpy()
        data[str(c)] = int(data[str(c)]) + int(cs)
print(data)"""

y = []
for data in train_dataloadersss:
    y.append(data[1])
print(torch.tensor([t[1].sum() for t in iter(train_dataloader)]).sum())
print(y)
""" noqa
# Get the length of class_names (one output unit for each class)
output_shape = len(class_names)

model = torchvision.models.efficientnet_b0(weights=weights).to(device)

# Recreate the classifier layer and seed it to the target device
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(
        in_features=1280,
        out_features=output_shape,
        bias=True,
    ),
).to(device)

# Freeze all base layers in the "features"
# section of the model (the feature extractor)
# by setting requires_grad=False
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
import os
import torch
import data_setup, engine, model_builder, utils
import torchvision

from torchvision import transforms
from pathlib import Path
from timeit import default_timer as timer
from torchinfo import summary
from utils import plot_loss_curves
from inference import pred_and_plot_image

# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Setup path to data folder
data_path = "/Users/shreyachauhan/Thesis_Document_Image_Classification/model/data" # noqa
image_path = os.path.join(data_path, "Tobacco3482_full")

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
        out_features=output_shape,  # same number of output units as our number of classes # noqa
        bias=True,
    ),
).to(device)

# Freeze all base layers in the "features" section of
# the model (the feature extractor) by
# setting requires_grad=False
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
results = engine.train(
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


# Plot the loss curves of our model
plot_loss_curves(results)
"""
