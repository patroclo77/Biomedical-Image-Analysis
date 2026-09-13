# We are ready to train the Cardiac Detection Model now!. In other words, we are going to predict where the heart is.

import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import cv2
from Common import *
from albumentations import BboxParams, Compose, Affine, RandomGamma

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


train_root_path = "Processed-Heart-Detection/train/"
train_subjects = "train_subjects.npy"
val_root_path = "Processed-Heart-Detection/val/"
val_subjects = "val_subjects.npy"

train_transforms = Compose(
    [
        RandomGamma(),   # closest equivalent to GammaContrast
        Affine(
            scale=(0.8, 1.2),
            rotate=(-10, 10),
            translate_px=(-10, 10)
        )
    ],
    bbox_params=BboxParams(format="pascal_voc", label_fields=["labels"])
)

train_dataset = CardiacDataSet("rsna_heart_detection.csv", train_subjects, train_root_path, train_transforms)
val_dataset   = CardiacDataSet("rsna_heart_detection.csv", val_subjects, val_root_path, None)

batch_size = 32
num_workers = 0

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size, num_workers=num_workers, shuffle=False)

model = CardiacDetectionModel()

checkpoint_callback = ModelCheckpoint(
    monitor="Val loss",
    save_top_k=10,
    mode="min",
)

trainer = pl.Trainer(accelerator="gpu", logger=TensorBoardLogger("logs/"), log_every_n_steps=1, callbacks=checkpoint_callback, max_epochs=100)

trainer.fit(model, train_loader, val_loader)