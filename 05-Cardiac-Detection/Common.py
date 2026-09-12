from pathlib import Path

import cv2
import torch
import numpy as np
import pandas as pd
#import imgaug
#from imgaug.augmentables.bbs import BoundingBox

import albumentations as A
import torchvision
from pyparsing import results
import pytorch_lightning as pl


class CardiacDataSet(torch.utils.data.Dataset):

    def __init__(self, path_to_labels_csv, patients, root_path, augs):
        
        self.labels = pd.read_csv(path_to_labels_csv)
        
        self.patients = np.load(patients)

        self.root_path = Path(root_path)

        self.augment = augs

    def  __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.patients)
        
    def __getitem__(self, idx):
        """
        Returns an image paired with bbox around the heart
        """
        patient = self.patients[idx]
        # Get data according to index
        data = self.labels[self.labels["name"]==patient]
        # Extract the patient_id (the filename)
        patient_id = data["name"].item()
        
        # Get entries of given patient
        # Extract coordinates
        bbox = []

        x_min = data["x0"].item()
        bbox.append(x_min)
        y_min = data["y0"].item()
        bbox.append(y_min)
        x_max = x_min + data["w"].item()  # get max from width
        bbox.append(x_max)
        y_max = y_min + data["h"].item()  # get max from height
        bbox.append(y_max)


        # Load file and convert to float32
        file_path = self.root_path/str(patient_id)  # Create the path to the file
        img = np.load(f"{file_path}.npy").astype(np.float32)
        
        
        # Apply imgaug augmentations to image and bounding box
        # Apply imgaug augmentations to image and bounding box
        # Apply albumentations augmentations to image and bounding box
        if self.augment:
            aug = self.augment(image=img,
                               bboxes=[bbox],
                               labels=[0])

            img = aug["image"]
            bbox = aug["bboxes"][0]

        # Normalize the image according to the values computed in Preprocessing
        img = (img - 0.494) / 0.252

        img = torch.tensor(img).unsqueeze(0)
        bbox = torch.tensor(bbox)
            
        return img, bbox


class CardiacDetectionModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # -- added for training -- #
        self.model = torchvision.models.resnet18(pretrained=True)

        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.model.fc = torch.nn.Linear(512, 4)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.loss_fn = torch.nn.MSELoss()


    def forward(self, data):
        return self.model(data)


    def training_step(self, batch, batch_idx):
        x_ray, label = batch
        label = label.float()
        pred = self(x_ray)
        loss = self.loss_fn(pred, label)

        self.log("Train loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx % 50 == 0:
            self.log_images(x_ray.cpu(), pred.cpu(), label.cpu(), "Train")

        return loss


    def validation_step(self, batch, batch_idx):
        x_ray, label = batch
        label = label.float()

        pred = self(x_ray)
        loss = self.loss_fn(pred, label)

        self.log("Val loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx % 50 == 0:
            self.log_images(x_ray.cpu(), pred.cpu(), label.cpu(), "Val")

        return loss


    def log_images(self, x_ray, pred, label, name):
        results = []

        for i in range(4):
            coords_labels = label[i]
            coords_pred   = pred[i]

            img = (x_ray[i] * 0.252 + 0.494).numpy()[0]

            x0, y0 = coords_labels[0].int().item(), coords_labels[1].int().item()
            x1, y1 = coords_labels[2].int().item(), coords_labels[3].int().item()

            img = cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 2)

            x0, y0 = coords_pred[0].int().item(), coords_pred[1].int().item()
            x1, y1 = coords_pred[2].int().item(), coords_pred[3].int().item()

            img = cv2.rectangle(img, (x0, y0), (x1, y1), (1, 1, 1), 2)

            results.append(torch.tensor(img).unsqueeze(0))

        grid = torchvision.utils.make_grid(torch.cat(results, dim=0), nrow=2)
        self.logger.experiment.add_image(name, grid, self.global_step)

    def configure_optimizers(self):
        return [self.optimizer]
