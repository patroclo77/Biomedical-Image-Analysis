# We are ready to train the Cardiac Detection Model now!. In other words, we are going to predict where the heart is.

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from Common import *
from albumentations import BboxParams, Compose, Affine, RandomGamma

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

train_root_path = "Processed-Heart-Detection/train/"
train_subjects = "Processed-Heart-Detection/train_subjects.npy"
val_root_path = "Processed-Heart-Detection/val/"
val_subjects = "Processed-Heart-Detection/val_subjects.npy"

train_transforms = Compose([
    RandomGamma(gamma_limit=(80, 120), p=1.0),  # approx. imgaug GammaContrast
    Affine(
        scale=(0.8, 1.2),
        rotate=(-10, 10),
        translate_percent=(-0.1, 0.1),  # or translate_px=(-10, 10) if you want pixel translation
        p=1.0
    )
], bbox_params=BboxParams(format='pascal_voc'))

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

trainer = pl.Trainer(accelerator="gpu",
                     logger=TensorBoardLogger("logs/"),
                     log_every_n_steps=1,
                     callbacks=checkpoint_callback,
                     max_steps=2500,
                     max_epochs=-1)

trainer.fit(model, train_loader, val_loader)

# tensorboard --logdir ./logs --port 6006