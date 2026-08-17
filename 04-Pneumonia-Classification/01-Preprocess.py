from pathlib import Path
import pydicom
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

labels = pd.read_csv("rsna-pneumonia-detection-challenge/stage_2_train_labels.csv")

labels.head(6)

labels = labels.drop_duplicates("patientId")

labels.head()

ROOT_PATH = Path("rsna-pneumonia-detection-challenge/stage_2_train_images")
SAVE_PATH = Path("Processed/")

fig, axis = plt.subplots(3, 3, figsize=(9, 9))

c = 0

for i in range(3):
    for j in range(3):
        patient_id = labels.patientId.iloc[c]
        dcm_path = ROOT_PATH/patient_id
        dcm_path = dcm_path.with_suffix(".dcm")
        dcm = pydicom.dcmread(dcm_path).pixel_array
        
        label = labels["Target"].iloc[c]
        
        axis[i][j].imshow(dcm, cmap="bone")
        axis[i][j].set_title(label)
        c+=1

sums = 0
sums_squared = 0

for c, patient_id in enumerate(tqdm(labels.patientId)):
    dcm_path = ROOT_PATH/patient_id  # Create the path to the dcm file
    dcm_path = dcm_path.with_suffix(".dcm")  # And add the .dcm suffix
    
    # Read the dicom file with pydicom and standardize the array
    dcm = pydicom.dcmread(dcm_path).pixel_array / 255
        
    # Resize the image as 1024x1024 is way to large to be handeled by Deep Learning models at the moment
    # Let's use a shape of 224x224
    # In order to use less space when storing the image we convert it to float16
    dcm_array = cv2.resize(dcm, (224, 224)).astype(np.float16)
    
    # Retrieve the corresponding label
    label = labels.Target.iloc[c]
    
    # 4/5 train split, 1/5 val split
    train_or_val = "train" if c < 24000 else "val" 
        
    current_save_path = SAVE_PATH/train_or_val/str(label) # Define save path and create if necessary
    current_save_path.mkdir(parents=True, exist_ok=True)
    np.save(current_save_path/patient_id, dcm_array)  # Save the array in the corresponding directory
    
    normalizer = dcm_array.shape[0] * dcm_array.shape[1]  # Normalize sum of image
    if train_or_val == "train":  # Only use train data to compute dataset statistics
        sums += np.sum(dcm_array) / normalizer
        sums_squared += (np.power(dcm_array, 2).sum()) / normalizer


from joblib import Parallel, delayed
import pydicom
import cv2
import numpy as np
from pathlib import Path

def process_one(c, patient_id):
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

results = Parallel(n_jobs=8)(delayed(process_one)(c, pid) for c, pid in enumerate(labels.patientId))

sums = sum(r[0] for r in results)
sums_squared = sum(r[1] for r in results)


mean = sums / 24000
std = np.sqrt(sums_squared / 24000 - (mean**2))

print(f"Mean of Dataset: {mean}, STD: {std}")
