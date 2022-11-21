from __future__ import annotations

import config as cfg
import data_setup_clean as data_setup

# Build data set using random shuffling
data_setup.build_dataset(
    datasetDir=cfg.TOBACCO_DATASET_PATH,
    trainDir=cfg.TRAIN_DIR,
    testDir=cfg.VAL_DIR,
    valSplit=cfg.VAL_SPLIT,
)

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
