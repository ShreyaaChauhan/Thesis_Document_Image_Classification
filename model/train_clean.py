from __future__ import annotations

import config as cfg
import data_setup_clean as data_setup
import torch
import torch.nn as nn
import torch.optim as optim
from engine import StepbyStep
from engine_new import Engine
import torchvision

# Build data set using random shuffling
# data_setup.build_dataset(
#     datasetDir=cfg.TOBACCO_DATASET_PATH,
#     trainDir=cfg.TRAIN_DIR,
#     testDir=cfg.VAL_DIR,
#     valSplit=cfg.VAL_SPLIT,
# )

# Generate DataLoader for both test and train Dataset
trainDataLoader, testDataLoader, classNames = data_setup.load_data(
    trainDir=cfg.TRAIN_DIR,
    testDir=cfg.VAL_DIR,
    trainTransforms=cfg.TRAIN_TRANSFORMS,
    valTransforms=cfg.VAL_TRANSFORMS,
    batch_size=cfg.BATCH_SIZE,
    imbalanced=cfg.IMBALANCED_DATASET,
    num_workers=cfg.NUM_WORKERS,
)

torch.manual_seed(13)
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
auto_transforms = weights.transforms()
model = torchvision.models.efficientnet_b0(weights=weights)
cfg.TRAIN_TRANSFORMS = weights.transforms()
cfg.VAL_TRANSFORMS = weights.transforms()

multi_loss_fn = nn.CrossEntropyLoss(reduction="mean")
optimizer_cnn1 = optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE)
ckpt_interval = 1

sbs_cnn1 = Engine(model, multi_loss_fn, optimizer_cnn1, ckpt_interval)
sbs_cnn1.set_loaders(trainDataLoader, testDataLoader)
sbs_cnn1.set_tensorboard("sbs_cnn1")
sbs_cnn1.train(cfg.EPOCHS)
sbs_cnn1.plot_losses()
print("calling visualize filter")
sbs_cnn1.visualize_filters("conv1", cmap="gray")
