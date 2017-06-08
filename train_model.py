from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications import VGG16
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout


def vgg_pp(x):
    x = x[:, :, ::-1]
    # Zero-center by mean pixel
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    return x

batch_size = 17
nb_train_samples = 136
nb_validation_samples = 17


model = VGG16(include_top=False, weights='imagenet')



datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=vgg_pp,
        fill_mode='nearest')

train_generator = datagen.flow_from_directory(
        'data/train',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode=None,  # this means our generator will only yield batches of data, no labels
        shuffle=False)

bottleneck_features_train = model.predict_generator(
        train_generator, nb_train_samples // batch_size)

np.save('bottleneck_features_train', bottleneck_features_train)


validation_generator = datagen.flow_from_directory(
        'data/validation',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode=None,  # this means our generator will only yield batches of data, no labels
        shuffle=False)

bottleneck_features_validation = model.predict_generator(
        validation_generator, nb_validation_samples // batch_size)

np.save('bottleneck_features_validation', bottleneck_features_validation)



train_data = np.load('bottleneck_features_train.npy')
train_labels = np.array(
    [0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))
print(train_labels.shape)

validation_data = np.load('bottleneck_features_validation.npy')
validation_labels = np.array(
    [0] * 9 + [1] * 8)
print(validation_labels.shape)
print(train_data.shape)

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_data, train_labels,
          epochs=30,
          batch_size=batch_size,
          validation_data=(validation_data, validation_labels))
## model.save_weights('top_fc_weights')
model.save('cm_pest')