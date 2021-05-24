import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from tqdm import tqdm
from keras.applications.resnet50 import preprocess_input, decode_predictions
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import ssl
import matplotlib.pyplot as plt
ssl._create_default_https_context = ssl._create_unverified_context

# define ResNet50 model
from Detect.AssessHumanFaceDetector import dog_files_short, human_files_short
ResNet50_mod = ResNet50(weights='imagenet')

def path_to_tensor(img_path):
    # загрузка RGB изображения как PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # преобразовать PIL.Image.Image тип в 3D-тензор с формой (224, 224, 3)
    x = image.img_to_array(img)
    # преобразовать 3D-тензор в 4D-тензор с формой (1, 224, 224, 3) и вернуть 4D-тензор
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def ResNet50_predict_labels(img_path):
    # возвращает вектор прогноза для изображения, расположенного в img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_mod.predict(img))

### возвращает "True", если собака обнаружена на изображении(по путю img_path)
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    plt.figure(figsize=(12, 12))
    plt.imshow(plt.imread(img_path))
    plt.show()
    return ((prediction <= 268) & (prediction >= 151))

### TODO: Проверка функции dog_detector
# accuracy_dog = (np.array(list(map(dog_detector, dog_files_short)))).mean()
# accuracy_human = (1-(np.array(list(map(dog_detector,human_files_short))))).mean()
# print('Точночть собак: {:.0f}%\nТочность людей: {:.0f}%'.format(accuracy_dog*100,accuracy_human*100))