import os

import torch
from torchvision import transforms

# specify path to the datasets
TOBACCO_DATASET_PATH = "/Users/shreyachauhan/Thesis_Document_Image_Classification/model/data/Tobacco3482"  # noqa

# specify the paths to train and val dataset
TRAIN_DIR = (
    "/Users/shreyachauhan/Thesis_Document_Image_Classification/model/data/Train"  # noqa
)
VAL_DIR = (
    "/Users/shreyachauhan/Thesis_Document_Image_Classification/model/data/Test"  # noqa
)
# set the input height and width
INPUT_HEIGHT = 224
INPUT_WIDTH = 224


# Set hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2
EPOCHS = 80

# set device to cuda if gpu is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# apply transformations to your data
TRAIN_TRANSFORMS = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(size=(INPUT_HEIGHT, INPUT_WIDTH)),
        # transforms.RandomHorizontalFlip(p=0.25),
        # transforms.RandomVerticalFlip(p=0.25),
        # transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229]),
    ],
)

VAL_TRANSFORMS = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(size=(INPUT_HEIGHT, INPUT_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229]),
    ],
)

IMBALANCED_DATASET = True

NUM_WORKERS = 0  # os.cpu_count()


CKPT_INTERVAL = 2

PRETRAINED = True

FINE_TUNE = True
