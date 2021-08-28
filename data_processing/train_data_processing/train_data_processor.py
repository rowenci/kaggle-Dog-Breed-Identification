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
    images = []
    for file_name in file_list:
        try:
            img = Image.open(file_name)
        except FileNotFoundError:
            print("file ont found")
            continue
        else:
            img = np.array(img)
            img = torch.from_numpy(img)
            img = img.transpose(0, 2)
            img = img.transpose(1, 2)
            images.append([file_name, img])
    return images

def loadLabels():
    labels = pd.read_csv("D:\Codes\AI\kaggle\kaggle-Dog-Breed-Identification\datas\images\labels.csv")
    print(labels)


loadLabels()