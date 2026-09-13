import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torch import Tensor

from Common import *


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CardiacDetectionModel.load_from_checkpoint("logs/lightning_logs/version_1/checkpoints/epoch=192-step=2500.ckpt")

model.eval()
model.to(device)

predictions = []
labels = []
val_root_path = "Processed-Heart-Detection/val/"
val_subjects = "val_subjects.npy"

val_dataset   = CardiacDataSet("rsna_heart_detection.csv", val_subjects, val_root_path, None)

with torch.no_grad():
    for data, label in val_dataset:
        data = data.to(device).float().unsqueeze(0)
        prediction = model(data)[0].cpu()

        predictions.append(prediction)
        labels.append(label)

predictions = torch.stack(predictions)
labels = torch.stack(labels)

print (abs(predictions - labels).mean(0))

IDX = 0
img, label = val_dataset[IDX]
prediction = predictions[IDX]

print (prediction)

fig, axis = plt.subplots(1, 1)
axis.imshow(img[0], cmap="bone")
heart = patches.Rectangle((prediction[0], prediction[1]), prediction[2] - prediction[0], prediction[3] - prediction[1], edgecolor="red", facecolor="none")
axis.add_patch(heart)
plt.tight_layout()
plt.show()

IDX = 10
img, label = val_dataset[IDX]
prediction = predictions[IDX]

print (prediction)

fig, axis = plt.subplots(1, 1)
axis.imshow(img[0], cmap="bone")
heart = patches.Rectangle((prediction[0], prediction[1]), prediction[2] - prediction[0], prediction[3] - prediction[1], edgecolor="red", facecolor="none")
axis.add_patch(heart)
plt.tight_layout()
plt.show()