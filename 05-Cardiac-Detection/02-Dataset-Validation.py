# In this notebook we will create a custom DataSet which will load
# and return an X-Ray image together with the location of the heart

# DataSet Creation
# Now we define the torch dataset! We need to define a __ len __ function which returns the length of the dataset and a __ getitem __ function which returns the image and corresponding bounding box.

# Additionally we apply data augmentation and normalization.

# Important: Configure Albumentations to transform the Pascal VOC bbox
# together with the image. CardiacDataSet passes the bbox via `bboxes`.

import Common
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from albumentations import BboxParams, Compose, Affine, RandomGamma

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

seq = Compose([
    RandomGamma(gamma_limit=(80, 120), p=1.0),  # approx. imgaug GammaContrast
    Affine(
        scale=(0.8, 1.2),
        rotate=(-10, 10),
        translate_percent=(-0.1, 0.1),  # or translate_px=(-10, 10) if you want pixel translation
        p=1.0
    )
], bbox_params=BboxParams(format='pascal_voc'))


labels_path = "./rsna_heart_detection.csv"
patients_path = "Processed-Heart-Detection/train_subjects.npy"
train_root = "Processed-Heart-Detection/train/"
dataset = Common.CardiacDataSet(labels_path, patients_path, train_root, seq)

img, bbox = dataset[0]

fig, axis = plt.subplots(1, 1)
axis.imshow(img[0], cmap="bone")
rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], edgecolor="r", facecolor="none")
axis.add_patch(rect)

plt.tight_layout()
plt.show()

# ------------------ Another Example -------------------- #

mg, label = dataset[17]

fig2, axis2 = plt.subplots(1, 1)
axis2.imshow(img[0], cmap="bone")
spot1 = patches.Rectangle((label[0], label[1]), label[2]-label[0], label[3]-label[1], edgecolor='r', facecolor='none')
axis2.add_patch(spot1)

axis2.set_title("X-RAY with BBOX around heart")
print(label)