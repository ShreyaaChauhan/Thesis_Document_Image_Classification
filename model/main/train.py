import config as cfg
import data_setup
import torch
import torch.nn as nn
import torch.optim as optim
from engine import Engine
import torchvision
from timeit import default_timer as timer

data_setup.build_dataset(
    datasetDir=cfg.TOBACCO_DATASET_PATH,
    trainDir=cfg.TRAIN_DIR,
    testDir=cfg.VAL_DIR,
    valSplit=cfg.VAL_SPLIT,
)

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
# weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
# auto_transforms = weights.transforms()
# model = torchvision.models.efficientnet_b0(weights=weights)

# weights = torchvision.models.AlexNet_Weights.DEFAULT
# auto_transforms = weights.transforms()


if cfg.PRETRAINED:
    print("[INFO]: Loading pre-trained weights")
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights)
    # auto_transforms = weights.transforms()
    # cfg.TRAIN_TRANSFORMS = weights.transforms()
    # print(auto_transforms)
    # model = torchvision.models.efficientnet_b0(weights=weights)
    # cfg.TRAIN_TRANSFORMS = weights.transforms()
    # cfg.VAL_TRANSFORMS = weights.transforms()
else:
    print("[INFO]: Not loading pre-trained weights")
    model = torchvision.models.efficientnet_b0()
if cfg.FINE_TUNE:
    print("[INFO]: Fine-tuning all layers...")
    for params in model.parameters():
        params.requires_grad = True
elif not cfg.FINE_TUNE:
    print("[INFO]: Freezing hidden layers...")
    for params in model.parameters():
        params.requires_grad = False
# Change the final classification head.
model.classifier[1] = nn.Linear(in_features=1280, out_features=output_shape).to(
    cfg.DEVICE
)


# weights = torchvision.models.EfficientNet_B4_Weights.DEFAULT
# auto_transforms = weights.transforms()
# print(auto_transforms)
# model = torchvision.models.efficientnet_b4(weights=weights)

# model.classifier = torch.nn.Sequential(
#     torch.nn.Dropout(p=0.4, inplace=True),
#     torch.nn.Linear(in_features=1792, out_features=output_shape, bias=True),
# ).to(cfg.DEVICE)
# print(model)


# # Here the size of each output sample is set to 2.
# # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
# model.fc = nn.Linear(num_ftrs, 10)


# # weights = torchvision.models.AlexNet_Weights.IMAGENET1K_V1
# # auto_transforms = weights.transforms()
# # print(auto_transforms)
# # model = torchvision.models.alexnet(weights=weights)
# # print(model)
# # model.classifier = torch.nn.Sequential(
# #     torch.nn.Dropout(p=0.2, inplace=True),
# #     torch.nn.Linear(
# #         in_features=9216,
# #         out_features=output_shape,
# #         bias=True,
# #     ),
# # ).to(cfg.DEVICE)


for param in model.features.parameters():
    param.requires_grad = False

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

sbs = Engine(model, loss_fn, optimizer, val_transforms=cfg.VAL_TRANSFORMS)
sbs.set_loader(trainDataLoader, testDataLoader)
inputs, classes = next(iter(trainDataLoader))
sbs.set_tensorboard("classy")
start_time = timer()
# sbs.load_checkpoint(
#     "/Users/shreyachauhan/Thesis_Document_Image_Classification/model/main/checkpoints/epoch_20.pth"
# )
sbs.train(n_epochs=cfg.EPOCHS)
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")
sbs.plot_losses()
sbs.plot_accuracy()
