import numpy as np
from PIL import ImageFile

from Detect.DetectDogs import paths_to_tensor
from Import.DogDataset import train_files, valid_files, test_files, train_targets, valid_targets, test_targets

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

from keras.callbacks import ModelCheckpoint

ImageFile.LOAD_TRUNCATED_IMAGES = True

# предварительно обработать данные для Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

model = Sequential()

model.add(Conv2D(input_shape=train_tensors.shape[1:],filters=16,kernel_size=2, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(filters=32,kernel_size=2, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(filters=64,kernel_size=2, activation='relu'))
model.add(MaxPooling2D())
model.add(GlobalAveragePooling2D())
model.add(Dense(133,activation='softmax'))

model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

### количество эпох для обучения модели
epochs = 20

checkpointer = ModelCheckpoint(filepath='../saved_models/weights.best.from_scratch.hdf5',
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets,
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)

model.load_weights('../saved_models/weights.best.from_scratch.hdf5')

# получить индекс прогнозируемой породы собаки для каждого изображения в тестовом наборе
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# отчет точности теста
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)