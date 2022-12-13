import config as cfg
import data_setup
import torch
import torch.nn as nn
import torch.optim as optim
from engine import Engine
import torchvision
from timeit import default_timer as timer

trainDataLoader, testDataLoader, classNames = data_setup.load_data(
    trainDir=cfg.TRAIN_DIR,
    testDir=cfg.VAL_DIR,
    trainTransforms=cfg.TRAIN_TRANSFORMS,
    valTransforms=cfg.VAL_TRANSFORMS,
    batch_size=cfg.BATCH_SIZE,
    imbalanced=cfg.IMBALANCED_DATASET,
    num_workers=cfg.NUM_WORKERS,
)
output_shape = len(classNames)
torch.manual_seed(13)
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
auto_transforms = weights.transforms()
model = torchvision.models.efficientnet_b0(weights=weights)
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(
        in_features=1280,
        out_features=output_shape,
        bias=True,
    ),
).to(cfg.DEVICE)
cfg.TRAIN_TRANSFORMS = weights.transforms()
cfg.VAL_TRANSFORMS = weights.transforms()
for param in model.features.parameters():
    param.requires_grad = False

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

sbs = Engine(model, loss_fn, optimizer)
sbs.set_loader(trainDataLoader, testDataLoader)
sbs.set_tensorboard("classy")
start_time = timer()
sbs.train(n_epochs=100)
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")
sbs.plot_losses()
sbs.plot_accuracy()
# sbs = engine(model, loss_fn, optimizer)
