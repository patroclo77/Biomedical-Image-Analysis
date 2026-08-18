import torch
import torchmetrics
import torchvision
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import Common

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Use strict=False, otherwise we would want to match the pos_weight which is not necessary
model = Common.PneumoniaModel.load_from_checkpoint("weights/sam_weights.ckpt")
model.eval()
model.to(device)

val_transforms = transforms.Compose([
                                    transforms.ToTensor(),  # Convert numpy array to tensor
                                    transforms.Normalize([0.49], [0.248]),  # Use mean and std from preprocessing notebook
])


val_dataset = torchvision.datasets.DatasetFolder(
    "Processed/val/",
    loader=Common.load_file,
    extensions=(".npy",),
    transform=val_transforms
)

preds = []
labels = []

with torch.no_grad():
    for data, label in tqdm(val_dataset):
        data = data.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")).float().unsqueeze(0)
        pred = torch.sigmoid(model(data)[0].cpu())
        preds.append(pred)
        labels.append(label)
preds  = torch.tensor(preds)
labels = torch.tensor(labels).int()



acc         = torchmetrics.Accuracy(task="binary")(preds, labels)
precision   = torchmetrics.Precision(task="binary")(preds, labels)
recall      = torchmetrics.Recall(task="binary")(preds, labels)
cm          = torchmetrics.ConfusionMatrix(task="binary")(preds, labels)
cm_threshed = torchmetrics.ConfusionMatrix(task="binary", threshold=0.25)(preds, labels)

print(f"Val Accuracy: {acc}")
print(f"Val Precision: {precision}")
print(f"Val Recall: {recall}")
print(f"Confusion Matrix:\n {cm}")
print(f"Confusion Matrix 2:\n {cm_threshed}")


fig, axis = plt.subplots(3, 3, figsize=(9, 9))

for i in range(3):
    for j in range(3):
        rnd_idx = np.random.randint(0, len(preds))
        axis[i][j].imshow(val_dataset[rnd_idx][0][0], cmap="bone")
        axis[i][j].set_title(f"Pred:{int(preds[rnd_idx] > 0.5)}, Label:{labels[rnd_idx]}")
        axis[i][j].axis("off")