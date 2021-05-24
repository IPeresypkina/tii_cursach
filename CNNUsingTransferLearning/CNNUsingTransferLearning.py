### TODO: Obtain bottleneck features from another pre-trained CNN.
import numpy as np
import matplotlib.pyplot as plt

from Detect.DetectDogs import path_to_tensor
from extract_bottleneck_features import *

from Import.DogDataset import train_files, valid_files, test_files, train_targets, valid_targets, test_targets, \
    dog_names

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

from keras.callbacks import ModelCheckpoint

bottleneck_features = np.load('../bottleneck_features/DogResnet50Data.npz')
train_ResNet50 = bottleneck_features['train']
valid_ResNet50 = bottleneck_features['valid']
test_ResNet50 = bottleneck_features['test']

ResNet50_model = Sequential()
ResNet50_model.add(GlobalAveragePooling2D(input_shape=train_ResNet50.shape[1:]))
ResNet50_model.add(Dense(133, activation='softmax'))

ResNet50_model.summary()
### Define the model


ResNet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
### Compile the model

checkpointer = ModelCheckpoint(filepath='../saved_models/weights.best.ResNet50.hdf5',
                               verbose=1, save_best_only=True)


history = ResNet50_model.fit(train_ResNet50, train_targets,
          validation_data=(valid_ResNet50, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

### Train the model.
ResNet50_model.load_weights('../saved_models/weights.best.ResNet50.hdf5')
### Load the model weights with the best validation loss.VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')


# get index of predicted dog breed for each image in test set
ResNet50_predictions = [np.argmax(ResNet50_model.predict(np.expand_dims(feature, axis=0))) for feature in test_ResNet50]

# report test accuracy
test_accuracy = 100*np.sum(np.array(ResNet50_predictions)==np.argmax(test_targets, axis=1))/len(ResNet50_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
### Calculate classification accuracy on the test dataset.

