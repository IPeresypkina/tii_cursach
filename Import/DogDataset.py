from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob

# функция для загрузки наборов данных (обучения, тестирвоания, проверки)
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# загрузка наборов данных
train_files, train_targets = load_dataset('../data/dogImages/train')
valid_files, valid_targets = load_dataset('../data/dogImages/valid')
test_files, test_targets = load_dataset('../data/dogImages/test')

# загружаем список пород
dog_names = [item[20:-1] for item in sorted(glob("../data/dogImages/train/*/"))]

# статистика о наборах данных
print('Всего %d категорий собак.' % len(dog_names))
print('Есть %s изображений собак.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('Есть %d тренировочных изображений собак.' % len(train_files))
print('Есть %d проверочных изображений собак.' % len(valid_files))
print('Есть %d тестовых изображений собак.'% len(test_files))