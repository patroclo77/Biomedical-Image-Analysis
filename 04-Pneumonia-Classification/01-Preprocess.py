import pydicom
import cv2
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
import Common
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Read csv containing the labels containing the disease

labels = pd.read_csv("rsna-pneumonia-detection-challenge/stage_2_train_labels.csv")

# Remove duplicates
labels = labels.drop_duplicates("patientId")
labels.head()

Common.print_example_images(labels)
plt.tight_layout()
plt.show()

# Dicom Reading & Effective storage
# In order to efficiently handle our data in the Dataloader, we convert the X-Ray images stored in the DICOM format to numpy arrays. Afterwards we compute the overall mean and standard deviation of the pixels of the whole dataset, for the purpose of normalization. Then the created numpy images are stored in two separate folders according to their binary label:
#
# 0: All X-Rays which do not show signs of pneumonia
# 1: All X-Rays which show signs of pneumonia
# To do so, we iterate over the patient ids and concat the patient ID with the ROOT_PATH.
#
# We then directly save the standardized and resized files into the corresponding directory (0 for healthy, 1 for pneumonia). This allows to take advantage of the ready-to-use torchvision DatasetFolder for simple file reading
#
# We standardize all images by the maximum pixel value in the provided dataset, 255. All images are resized to 224x224.
#
# To compute dataset mean and standard deviation, we compute the sum of the pixel values as well as the sum of the squared pixel values for each subject. This allows to compute the overall mean and standard deviation without keeping the whole dataset in memory.
#
# We will use mean and std later in the dat

sums = 0
sums_squared = 0

for c, patient_id in enumerate(tqdm(labels.patientId)):
    dcm_path = Common.ROOT_PATH/patient_id  # Create the path to the dcm file
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
        
    current_save_path = Common.SAVE_PATH/train_or_val/str(label) # Define save path and create if necessary

    current_save_path.mkdir(parents=True, exist_ok=True)

    np.save(current_save_path/patient_id, dcm_array)  # Save the array in the corresponding directory
    
    normalizer = dcm_array.shape[0] * dcm_array.shape[1]  # Normalize sum of image

    if train_or_val == "train":  # Only use train data to compute dataset statistics
        sums += np.sum(dcm_array) / normalizer
        sums_squared += (np.power(dcm_array, 2).sum()) / normalizer


results = Parallel(n_jobs=8)(
    delayed(Common.process_one_patient)(labels, c, pid)
    for c, pid in tqdm(enumerate(labels.patientId), total=len(labels))
)

sums         = sum(r[0] for r in results)
sums_squared = sum(r[1] for r in results)

mean = sums / 24000
std  = np.sqrt(sums_squared / 24000 - (mean**2))

print(f"Mean of Dataset: {mean}, STD: {std}")
