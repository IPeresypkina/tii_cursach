import cv2
# extract pre-trained face detector
import numpy as np
from Import.DogDataset import train_files
from Import.ImportHuman import human_files

face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalcatface.xml')

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

human_files_short = human_files[:100]
dog_files_short = train_files[:100]
# Do NOT modify the code above this line.
## TODO: Test the performance of the face_detector algorithm
## on the images in human_files_short and dog_files_short.
accuracy_dog = 100-np.array(list(map(face_detector, dog_files_short))).sum()
accuracy_human = np.array(list(map(face_detector, human_files_short))).sum()

print('Точность распознования собак: {}% Точность распознавания людей: {}%'.format(accuracy_dog, accuracy_human))