import torch
import torchvision
from torchvision import transforms
import pytorch_lightning as ply
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import matplotlib.pyplot as plt
import Common

# First we create our dataset. We can leverage the DatasetFolder from torchvision: It allows to simply pass a root directory and return return a dataset object with access to all files within the directory and the directory name as class label.
# We only need to define a loader function, load_file, which defines how the files shall be loaded. This is very comfortable as we only have to load our previously stored numpy files. Additionally, we need to define a list of file extensions (just "npy" in our case).
#
# Finally, we can pass a transformation sequence for Data Augmentation and Normalization.
#
# We use:
#
# - RandomResizedCrops which applies a random crop of the image and resizes it to the original image size (224x224)
# - Random Rotations between -5 and 5 degrees
# - Random Translation (max 5%)
# - Random Scaling (0.9-1.1 of original image size)

train_transforms = transforms.Compose([
                                    transforms.ToTensor(),  # Convert numpy array to tensor
                                    transforms.Normalize(0.49, 0.248),  # Use mean and std from preprocessing notebook
                                    transforms.RandomAffine( # Data Augmentation
                                        degrees=(-5, 5), translate=(0, 0.05), scale=(0.9, 1.1)),
                                        transforms.RandomResizedCrop((224, 224), scale=(0.35, 1))

])

val_transforms = transforms.Compose([
                                    transforms.ToTensor(),  # Convert numpy array to tensor
                                    transforms.Normalize([0.49], [0.248]),  # Use mean and std from preprocessing notebook
])


train_dataset = torchvision.datasets.DatasetFolder(
    "Processed/train/",
    loader=Common.load_file,
    extensions=(".npy",),
    transform=train_transforms
)

val_dataset = torchvision.datasets.DatasetFolder(
    "Processed/val/",
    loader=Common.load_file,
    extensions=(".npy",),
    transform=val_transforms
)

Common.plot_examples_augmented_train_images(train_dataset)
plt.tight_layout()
plt.show()

# Finally, we create the train and val dataset and the corresponding data loaders.

# Please adapt batch size and num_workers according to your hardware resources.

batch_size  = 32
num_workers = 0

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size, num_workers=num_workers, shuffle=False)

print(f"There are {len(train_dataset)} train images and {len(val_dataset)} val images")

# The classes are imbalanced: There are more images without signs of pneumonia than with pneumonia. There are multiple ways to deal with imbalanced datasets:
#
# - Weighted Loss
# - Oversampling
# - Doing nothing :)
# In this example, we will simply do nothing as this often yields the best results. Buf feel free to play around with a weighted loss. A template to define a customized weighted loss function is provided below.

np.unique(train_dataset.targets, return_counts=True), np.unique(val_dataset.targets, return_counts=True)

model = Common.PneumoniaModel()

# Create the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='Val Acc',
    save_top_k=10,
    mode='max')

torch.set_float32_matmul_precision("high")

trainer = ply.Trainer(accelerator="gpu",
                     logger=TensorBoardLogger(save_dir="./logs"),
                     log_every_n_steps=1,
                     callbacks=checkpoint_callback,
                     max_epochs=35)

trainer.fit(model, train_loader, val_loader)

trainer.save_checkpoint("weights/sam_weights.ckpt")

