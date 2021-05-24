import cv2
import fire
import numpy as np
from tensorflow.python.keras.models import load_model
from Detect.AssessPetsFaceDetector import cat_detector
from Detect.DetectDogs import dog_detector, path_to_tensor
from Import.DogDataset import dog_names
from extract_bottleneck_features import extract_Resnet50

class Detect:
    def classify_dog_breed(img_path):
        img = path_to_tensor(img_path)
        predictions = new_model.predict(extract_Resnet50(img))
        prediction = np.argmax(predictions)
        breed = dog_names[prediction].split('.')[-1]
        markBreed = dog_names[prediction].split('.')[0]
        # for predict in predictions:
        print('Собака породы {}. {:2.0f}%. mark {}'.format(breed,  100 * np.max(predictions), markBreed.split('/')[1]))
        return breed

    def return_type_pet(img_path):
        if cat_detector(img_path):
            print('Cat Detected')
            print('\n')
            return 'Cat'
        elif dog_detector(img_path):
            print('Dog Detected')
            print('\n')
            return 'Dog'
        else:
            print('Neither cat nor Dog detected')
            print('\n')
            return 0

    def return_breed(img_path, type):
        if type == 'Cat':
            print('Cat Detected')
            print('\n')
            return 'Cat'
        elif type == 'Dog':
            print('Dog Detected')
            breed = Detect.classify_dog_breed(img_path)
            print('\n')
            return breed
        else:
            return 0

# подгружаем обученную модель
new_model = load_model('../saved_models/weights.best.ResNet50.hdf5')
# Проверим ее архитектуру
# new_model.summary()
img_path = '../images/human.jpeg'
# img_path = argv[1]
img = cv2.imread(img_path)
# def return_pet(img_path, type, breed):
# convert BGR image to RGB for plotting
# cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
type_pet = Detect.return_type_pet(img_path)
breed_pet = Detect.return_breed(img_path, type_pet)

if __name__ == '__main__':
  fire.Fire(Detect)
