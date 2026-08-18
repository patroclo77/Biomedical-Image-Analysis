import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import Common
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

val_transforms = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(0.49, 0.248)
])

val_dataset = torchvision.datasets.DatasetFolder("Processed/val/", loader=Common.load_file, extensions='npy', transform=val_transforms)

temp_model = torchvision.models.resnet18()

torch.nn.Sequential(*list(temp_model.children())[:-2])

model = Common.PneumoniaModel.load_from_checkpoint("weights/sam_weights.ckpt", strict=False)
model.eval()

device = next(model.parameters()).device

print(device)

img = val_dataset[-6][0]  # Select a subject
img = img.to(device)

target_layer = model.model.layer4[-1]

gradcam = Common.GradCAM(model, target_layer)

cam_img, pred_gradcam = gradcam(img)
Common.visualize(img.cpu(), cam_img, pred_gradcam)
plt.tight_layout()
plt.show()

