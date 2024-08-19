import PIL
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import sys
import os, cv2
import csv
import pandas as pd
myDir = "..\GujOCR\Output"

def createFileList(myDir, format='.png'):
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList
columnNames = list()
for i in range(784):
    pixel = 'p'
    pixel += str(i)
    columnNames.append(pixel)
l = os.listdir("..\GujOCR\Output")
print(l)

dic = {val : idx for idx, val in enumerate(l)}
print(dic)

train_data = pd.DataFrame(columns = columnNames)
train_data.to_csv("trainset28.csv",index = False)
label_count = list()

print(len(l))

for i in range(len(l)):
    mydir = 'OUTPUT/' + l[i]
    fileList = createFileList(mydir)
    for file in fileList:
        img_file = Image.open(file)
        width, height = img_file.size
        format = img_file.format
        mode = img_file.mode

        label_count.append(dic[l[i]])
        inverted_image = img_file.convert('RGB')
        im_invert = ImageOps.invert(inverted_image)
        size = (28, 28)
        new_image = img_file.resize(size)
        enhancer = ImageEnhance.Contrast(new_image)
        new_image = enhancer.enhance(3)
        img_grey = new_image.convert('L')
        value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))
        value = value.flatten()
        with open("trainset28.csv", 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(value)

read_data = pd.read_csv('trainset28.csv')
read_data['Label'] = label_count
print(read_data)

read_data.to_csv("training_label28.csv",index = False)
print(train_data)
