#%%
import torch
import torchvision
import matplotlib.pyplot as plt
from torchinfo import summary
import helper_functions as hlp_fn
import config as cfg
import data_setup_clean as data_setup

print(f"torch version: {torch.__version__}")
print(f"torchvision version: {torchvision.__version__}")

print(cfg.DEVICE)

trainDataLoader, testDataLoader, classNames = data_setup.load_data(
    trainDir=cfg.TRAIN_DIR,
    testDir=cfg.VAL_DIR,
    trainTransforms=cfg.TRAIN_TRANSFORMS,
    valTransforms=cfg.VAL_TRANSFORMS,
    batch_size=cfg.BATCH_SIZE,
    imbalanced=cfg.IMBALANCED_DATASET,
    num_workers=cfg.NUM_WORKERS,
)

# Get a batch of images
image_batch, label_batch = next(iter(trainDataLoader))

# Get a single image from the batch
image, label = image_batch[0], label_batch[0]

# View the batch shapes
print(image.shape, label)
plt.imshow(
    image.permute(1, 2, 0)
)  # rearrange image dimensions to suit matplotlib [color_channels, height, width] -> [height, width, color_channels]
plt.title(classNames[label])
plt.axis(False)
plt.savefig("abc.png")

# %%
