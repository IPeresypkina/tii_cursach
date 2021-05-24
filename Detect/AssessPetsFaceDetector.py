import cv2
# extract pre-trained face detector
import numpy as np
from Import.DogDataset import train_files
from Import.ImportCat import cat_files

cat_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalcatface.xml')

def cat_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cats = cat_cascade.detectMultiScale(gray)
    return len(cats) > 0

# returns "True" if face is detected in image stored at img_path
# def face_detector(img_path):
#     img = cv2.imread(img_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray)
#     return len(faces) > 0

dog_files_short = train_files[:100]
cat_files_short = cat_files[:100]
# Do NOT modify the code above this line.
## TODO: Test the performance of the face_detector algorithm
## on the images in human_files_short and dog_files_short.
accuracy_dog = 100-np.array(list(map(cat_detector, dog_files_short))).sum()
accuracy_cat = np.array(list(map(cat_detector, cat_files_short))).sum()

print('Точность распознования собак: {}% Точность распознавания котов: {}%'.format(accuracy_dog, accuracy_cat))