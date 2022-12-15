import os
import shutil
from collections import Counter
import numpy as np
import torch
from imutils import paths
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
import config as cfg


def copy_images(image_Paths: list, folder: str):
    shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)
    for path in image_Paths:
        imageName = path.split(os.path.sep)[-1]
        label = path.split(os.path.sep)[-2]
        labelFolder = os.path.join(folder, label)

        os.makedirs(labelFolder, exist_ok=True)

        destination = os.path.join(labelFolder, imageName)
        shutil.copy(path, destination)


def make_balanced_sampler(y):
    y = torch.as_tensor(y)
    classes, counts = y.unique(return_counts=True)
    weights = 1.0 / counts.float()
    sample_weights = weights[y.squeeze().long()]
    generator = torch.Generator()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        generator=generator,
        replacement=True,
    )
    return sampler


def print_class_image_count(dataLoader):
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
    for t in iter(dataLoader):
        classes, counts = t[1].unique(return_counts=True)
        for c, cs in zip(classes, counts):
            c = c.cpu().detach().numpy()
            cs = cs.cpu().detach().numpy()
            data[str(c)] = int(data[str(c)]) + int(cs)
    return data


def build_dataset(
    datasetDir: str,
    trainDir: str,
    testDir: str,
    valSplit: float,
):  # noqa
    print("[[[INFO]Loading image paths...")
    imagePaths = list(paths.list_images(datasetDir))
    os.makedirs(trainDir, exist_ok=True)
    os.makedirs(testDir, exist_ok=True)
    np.random.shuffle(imagePaths)
    valImagesLen = int(len(imagePaths) * valSplit)
    trainImagesLen = len(imagePaths) - valImagesLen
    trainPaths = imagePaths[:trainImagesLen]
    valPaths = imagePaths[trainImagesLen:]
    print("[[INFO]Copying Image from sorce folder to destination...")
    copy_images(trainPaths, trainDir)
    copy_images(valPaths, testDir)
    print(
        f"[[INFO] {len(list(paths.list_images(trainDir)))} \
    images in TRAIN folder...",
    )
    print(
        f"[[INFO] {len(list(paths.list_images(testDir)))} \
    images in Test folder...",
    )


def load_data(
    trainDir: str,
    testDir: str,
    trainTransforms: transforms.Compose,
    valTransforms: transforms.Compose,
    batch_size: int,
    imbalanced: bool,
    num_workers: int,
):
    print("[[INFO] Loading the training and validation dataset...")
    trainDataset = ImageFolder(root=trainDir, transform=trainTransforms)
    valDataset = ImageFolder(root=testDir, transform=valTransforms)

    print(f"[[INFO] Training dataset contains {len(trainDataset)} samples...")
    print(f"[[INFO] Validation dataset contains {len(valDataset)} samples...")

    # get class names
    class_names = trainDataset.classes
    print(f"[[INFO] Dataset contains following classes \n[[INFO]  {class_names}")
    print(f"[[INFO] {dict(Counter(trainDataset.targets))}")
    print("[[INFO] Creating training and validation set dataloaders...")
    if imbalanced:
        labels = [data[1] for data in trainDataset]
        sampler = make_balanced_sampler(labels)
        trainDataLoader = DataLoader(
            trainDataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=True,
        )
        print("[[INFO] After Sampling Imbalanced Dataset")
        print(f"[[INFO] {print_class_image_count(trainDataLoader)}")

    else:
        trainDataLoader = DataLoader(
            trainDataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
    valDataLoader = DataLoader(
        valDataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    return (
        trainDataLoader,
        valDataLoader,
        class_names,
    )


def load_complete_val_dataset(testDir: str, valTransforms: transforms.Compose):
    valDataset = ImageFolder(root=testDir, transform=valTransforms)
    class_names = valDataset.classes
    batch = len(valDataset)
    valDataLoader = DataLoader(valDataset, batch_size=batch)
    return valDataLoader, class_names, batch
