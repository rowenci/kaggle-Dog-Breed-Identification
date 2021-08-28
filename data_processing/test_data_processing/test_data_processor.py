import torch
import numpy as np
import pandas as pd
from PIL import Image

import os

path = "D:\Codes\AI\kaggle\kaggle-Dog-Breed-Identification\datas\images\\test"
# get all image files' names
# file_number = 10357
for imageName in os.listdir(path):
    # print(os.path.join(path, imageName))
    pass