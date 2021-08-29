import torch
import numpy as np
import pandas as pd
from PIL import Image

import os

path = "D:\Codes\AI\kaggle\kaggle-Dog-Breed-Identification\datas\images\\train"
# get all image files' names
# file_number = 10222
file_list = []
for imageName in os.listdir(path):
    file_list.append(os.path.join(path, imageName))

# iteratly open files
def loadImages():
    images = {}
    for file_path in file_list:
        try:
            img = Image.open(file_path)
            file_name_full = file_path.split("\\")[-1]
            file_name = file_name_full.split(".")[0]
        except FileNotFoundError:
            print("file ont found")
            continue
        else:
            img = np.array(img)
            img = torch.from_numpy(img)
            img = img.transpose(0, 2)
            img = img.transpose(1, 2)
            images[file_name] = img
    return images

def loadLabels():
    labels = pd.read_csv("D:\Codes\AI\kaggle\kaggle-Dog-Breed-Identification\datas\images\labels.csv")
    labels["image"] = np.nan
    return labels


images = loadImages()
labels = loadLabels()
"""
for idx in labels["id"]:
    image_value = images[idx]
    print(image_value)
    labels.loc[labels["id"] == "000bec180eb18c7604dcecc8fe0dba07"]["image"] = image_value
print(labels)
"""
