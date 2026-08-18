from pathlib import Path
import numpy as np
import pydicom
import cv2
import matplotlib.pyplot as plt
import torchmetrics
import torchvision
import pytorch_lightning as ply
from torchvision import transforms
import torch
import torch.nn.functional as F

# ---------------- BEGIN 01-Preprocess ---------------- #

# DCOM files path and our processed files

ROOT_PATH = Path("rsna-pneumonia-detection-challenge/stage_2_train_images")
SAVE_PATH = Path("Processed/")

def process_one_patient(labels, c, patient_id):
    dcm_path = (ROOT_PATH / patient_id).with_suffix(".dcm")
    dcm = pydicom.dcmread(dcm_path).pixel_array / 255
    dcm_array = cv2.resize(dcm, (224, 224)).astype(np.float16)

    label = labels.Target.iloc[c]
    train_or_val = "train" if c < 24000 else "val"
    current_save_path = SAVE_PATH / train_or_val / str(label)
    current_save_path.mkdir(parents=True, exist_ok=True)
    np.save(current_save_path / patient_id, dcm_array)

    normalizer = dcm_array.size

    return (
        np.sum(dcm_array) / normalizer if train_or_val == "train" else 0,
        np.sum(dcm_array ** 2) / normalizer if train_or_val == "train" else 0
    )


def print_example_images(labels):

    fig, axis = plt.subplots(3, 3, figsize=(9, 9))

    c = 0

    for i in range(3):
        for j in range(3):
            patient_id = labels.patientId.iloc[c]
            dcm_path = ROOT_PATH / patient_id
            dcm_path = dcm_path.with_suffix(".dcm")
            dcm = pydicom.dcmread(dcm_path).pixel_array

            label = labels["Target"].iloc[c]

            axis[i][j].imshow(dcm, cmap="bone")
            axis[i][j].set_title(label)
            c += 1

# ---------------- END 01-Preprocess ---------------- #
# ---------------- BEGIN 02-Train ---------------- #

def load_file(path):
    return np.load(path).astype(np.float32)

def plot_examples_augmented_train_images(train_dataset):
    indexes = [4847, 673, 19712, 10728]
    k = 0
    fig, axis = plt.subplots(2, 2, figsize=(9, 9))
    for i in range(2):
        for j in range(2):
            x_ray, label = train_dataset[indexes[k]]
            axis[i][j].imshow(x_ray[0], cmap="bone")
            axis[i][j].set_title(f"Label:{label}")
            k += 1
# Optimizer and Loss
# We use the Adam Optimizer with a learning rate of 0.0001 and the BinaryCrossEntropy Loss function.
# (In fact we use BCEWithLogitsLoss which directly accepts the raw unprocessed predicted values and computes
# the sigmoid activation function before applying Cross Entropy). Feel free to pass a weight different
# from 1 to the Pneumonia model in order to use the weighted loss function.

class PneumoniaModel(ply.LightningModule):
    def __init__(self, weight=1):
        super().__init__()

        self.model = torchvision.models.resnet18()
        # change conv1 from 3 to 1 input channels
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # change out_feature of the last fully connected layer (called fc in resnet18) from 1000 to 1
        self.model.fc = torch.nn.Linear(in_features=512, out_features=1)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]))

        # simple accuracy computation
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_acc   = torchmetrics.Accuracy(task="multiclass", num_classes=2)

        self.feature_map = torch.nn.Sequential(*list(self.model.children())[:-2])

    def forward(self, data):
        feature_map = self.feature_map(data)

        # Adaptive pool → (batch, 512, 1, 1)
        avg_pool_output = torch.nn.functional.adaptive_avg_pool2d(feature_map, (1, 1))

        # Flatten only feature dims → (batch, 512)
        avg_pool_output_flattened = torch.flatten(avg_pool_output, start_dim=1)

        # Final prediction
        pred = self.model.fc(avg_pool_output_flattened)

        return pred, feature_map

    def training_step(self, batch, batch_idx):
        x_ray, label = batch
        label = label.float()

        pred, _ = self(x_ray)  # unpack
        pred = pred[:, 0]  # now safe

        loss = self.loss_fn(pred, label)

        self.log("Train Loss", loss)
        self.log("Step Train Acc", self.train_acc(torch.sigmoid(pred), label.int()))
        return loss

    def on_train_epoch_end(self):
        # After one epoch compute the whole train_data accuracy
        self.log("Train Acc", self.train_acc.compute())

    def validation_step(self, batch, batch_idx):
        x_ray, label = batch
        label = label.float()

        pred, _ = self(x_ray)  # unpack the tuple
        pred = pred[:, 0]  # now safe

        loss = self.loss_fn(pred, label)

        self.log("Val Loss", loss)
        self.log("Step Val Acc", self.val_acc(torch.sigmoid(pred), label.int()))
        return loss

    def on_validation_epoch_end(self):
        self.log("Val Acc", self.val_acc.compute())

    def configure_optimizers(self):
        # Caution! You always need to return a list here (just pack your optimizer into one :))
        return [self.optimizer]

# ---------------- END 02-Train ---------------- #

# ---------------- BEGIN 03-Interpretability ---------------- #

def cam(model, img):
    """
    Compute class activation map according to cam algorithm
    """
    with torch.no_grad():
        pred, features = model(img.unsqueeze(0))
    b, c, h, w = features.shape

    # We reshape the 512x7x7 feature tensor into a 512x49 tensor in order to simplify the multiplication
    features = features.reshape((c, h * w))

    # Get only the weights, not the bias
    weight_params = list(model.model.fc.parameters())[0]

    # Remove gradient information from weight parameters to enable numpy conversion
    weight = weight_params[0].detach()
    print(weight.shape)
    # Compute multiplication between weight and features with the formula from above.
    # We use matmul because it directly multiplies each filter with the weights
    # and then computes the sum. This yields a vector of 49 (7x7 elements)
    cam = torch.matmul(weight, features)
    print(features.shape)

    ### The following loop performs the same operations in a less optimized way
    # cam = torch.zeros((7 * 7))
    # for i in range(len(cam)):
    #    cam[i] = torch.sum(weight*features[:,i])
    ##################################################################

    # Normalize and standardize the class activation map (Not always necessary, thus not shown in the lecture)
    cam = cam - torch.min(cam)
    cam_img = cam / torch.max(cam)
    # Reshape the class activation map to 512x7x7 and move the tensor back to CPU
    cam_img = cam_img.reshape(h, w).cpu()

    return cam_img, torch.sigmoid(pred)


def visualize(img, cam, pred):
    # Ensure no gradients
    img = img.detach().cpu()
    cam = cam.detach().cpu()

    # Resize both to same size
    img = transforms.functional.resize(img, (224, 224))
    cam = transforms.functional.resize(cam.unsqueeze(0), (224, 224))[0]

    # Convert grayscale (1,H,W) → (H,W)
    img = img[0]

    # Convert to numpy NOW (critical)
    img = img.numpy()
    cam = cam.numpy()

    fig, axis = plt.subplots(1, 2, figsize=(8, 4))

    axis[0].imshow(img, cmap="bone")
    axis[1].imshow(img, cmap="bone")
    axis[1].imshow(cam, alpha=0.5, cmap="jet")

    axis[1].set_title(f"Pneumonia: {(pred > 0.5).item()}")

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()

        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward hook
        target_layer.register_forward_hook(self.save_activation)

        # Register backward hook
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, img):
        """

        :rtype: tuple[Any, Tensor]
        """
        device = next(self.model.parameters()).device
        img = img.to(device).unsqueeze(0)

        # Forward pass
        pred, _ = self.model(img)
        pred = pred.to(device)   # <-- ensure pred is on GPU

        pred = pred.view(-1)  # always 1‑D
        pred.backward(torch.ones_like(pred))

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        cam = F.relu(cam)
        cam = cam.squeeze().cpu()

        # Normalize
        cam -= cam.min()
        cam /= cam.max()

        return cam, torch.sigmoid(pred.cpu())
