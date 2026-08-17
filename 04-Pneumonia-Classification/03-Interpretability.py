import torch
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



def load_file(path):
    return np.load(path).astype(np.float32)


val_transforms = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(0.49, 0.248),

])

val_dataset = torchvision.datasets.DatasetFolder("Processed/val/", loader=load_file, extensions="npy", transform=val_transforms)

temp_model = torchvision.models.resnet18()
temp_model

list(temp_model.children())[:-2]  # get all layers up to avgpool


torch.nn.Sequential(*list(temp_model.children())[:-2])


class PneumoniaModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = torchvision.models.resnet18()
        # Change conv1 from 3 to 1 input channels
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Change out_feature of the last fully connected layer (called fc in resnet18) from 1000 to 1
        self.model.fc = torch.nn.Linear(in_features=512, out_features=1)

        # Extract the feature map
        self.feature_map = torch.nn.Sequential(*list(self.model.children())[:-2])

    def forward(self, data):
        # Compute feature map
        feature_map = self.feature_map(data)
        # Use Adaptive Average Pooling as in the original model
        avg_pool_output = torch.nn.functional.adaptive_avg_pool2d(input=feature_map, output_size=(1, 1))
        print(avg_pool_output.shape)
        # Flatten the output into a 512 element vector
        avg_pool_output_flattened = torch.flatten(avg_pool_output)
        print(avg_pool_output_flattened.shape)
        # Compute prediction
        pred = self.model.fc(avg_pool_output_flattened)
        return pred, feature_map


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

model = PneumoniaModel.load_from_checkpoint("weights/sam_weights.ckpt", strict=False)
model.eval()

device = next(model.parameters()).device
img = val_dataset[-6][0]  # Select a subject
img = img.to(device)
activation_map, pred = cam(model, img)

import torch
import torch.nn.functional as F

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
        device = next(self.model.parameters()).device
        img = img.to(device).unsqueeze(0)

        # Forward pass
        pred, _ = self.model(img)
        pred = pred.to(device)   # <-- ensure pred is on GPU


        # Backward pass for class 1 (pneumonia)
        self.model.zero_grad()
        pred.backward(torch.tensor([1.0], device=device))

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        cam = F.relu(cam)
        cam = cam.squeeze().cpu()

        # Normalize
        cam -= cam.min()
        cam /= cam.max()

        return cam, torch.sigmoid(pred.cpu())

target_layer = model.model.layer4[-1]

gradcam = GradCAM(model, target_layer)

cam_img, pred = gradcam(img)

print("Plotting img")

visualize(img.cpu(), cam_img, pred)
plt.tight_layout()
plt.show()

print("Plotting img finished")

