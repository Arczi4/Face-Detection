import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import pickle

Path = './faces/'
Files = [f for f in listdir(Path) if isfile(join(Path, f))]

Training_Data, Labels = [], []

for i, file in enumerate(Files):
    img_path = Path + Files[i]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(img, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_Data), np.asarray(Labels))

filename = 'finalized_model.sav'
model.write(filename)

print("Success")
