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
"""
[image name, tensor]
"""
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

"""
[index, id, label, image]
"""
def loadLabels():
    labels = pd.read_csv("D:\Codes\AI\kaggle\kaggle-Dog-Breed-Identification\datas\images\labels.csv")
    return labels


images = loadImages()
labels = loadLabels()
imgs = []
# 将图片tensor以label的顺序进行存储
for label in labels["id"]:
    imgs.append(images[label])
imgs = np.array(imgs)
# 将狗的品种换成0 ~ 119的数字进行存储
breeds = '''affenpinscher
afghan_hound
african_hunting_dog
airedale
american_staffordshire_terrier
appenzeller
australian_terrier
basenji
basset
beagle
bedlington_terrier
bernese_mountain_dog
black-and-tan_coonhound
blenheim_spaniel
bloodhound
bluetick
border_collie
border_terrier
borzoi
boston_bull
bouvier_des_flandres
boxer
brabancon_griffon
briard
brittany_spaniel
bull_mastiff
cairn
cardigan
chesapeake_bay_retriever
chihuahua
chow
clumber
cocker_spaniel
collie
curly-coated_retriever
dandie_dinmont
dhole
dingo
doberman
english_foxhound
english_setter
english_springer
entlebucher
eskimo_dog
flat-coated_retriever
french_bulldog
german_shepherd
german_short-haired_pointer
giant_schnauzer
golden_retriever
gordon_setter
great_dane
great_pyrenees
greater_swiss_mountain_dog
groenendael
ibizan_hound
irish_setter
irish_terrier
irish_water_spaniel
irish_wolfhound
italian_greyhound
japanese_spaniel
keeshond
kelpie
kerry_blue_terrier
komondor
kuvasz
labrador_retriever
lakeland_terrier
leonberg
lhasa
malamute
malinois
maltese_dog
mexican_hairless
miniature_pinscher
miniature_poodle
miniature_schnauzer
newfoundland
norfolk_terrier
norwegian_elkhound
norwich_terrier
old_english_sheepdog
otterhound
papillon
pekinese
pembroke
pomeranian
pug
redbone
rhodesian_ridgeback
rottweiler
saint_bernard
saluki
samoyed
schipperke
scotch_terrier
scottish_deerhound
sealyham_terrier
shetland_sheepdog
shih-tzu
siberian_husky
silky_terrier
soft-coated_wheaten_terrier
staffordshire_bullterrier
standard_poodle
standard_schnauzer
sussex_spaniel
tibetan_mastiff
tibetan_terrier
toy_poodle
toy_terrier
vizsla
walker_hound
weimaraner
welsh_springer_spaniel
west_highland_white_terrier
whippet
wire-haired_fox_terrier
yorkshire_terrier
'''
breeds = breeds.split('\n')
breeds_dict = {}
idx = 0
for breed in breeds:
    breeds_dict[breed] = idx
    idx += 1
breeds_dict.popitem()
# 按labels顺序来将breed变成对应的数字
label_list = np.ndarray(10222)
i = 0
for i in range(10222):
    label_list[i] = breeds_dict[labels.iloc[i, 1]]
np.save("D:\Codes\AI\kaggle\kaggle-Dog-Breed-Identification\datas\pixeled_images\images.npy", imgs)
np.save("D:\Codes\AI\kaggle\kaggle-Dog-Breed-Identification\datas\pixeled_images\labels.npy", label_list)