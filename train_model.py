from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16

import numpy as np

from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential

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
batch_num_train = nb_train_samples // batch_size * 10
batch_num_validation = nb_validation_samples // batch_size * 10

#
# model = VGG16(include_top=False, weights='imagenet')
#
#
# datagen = ImageDataGenerator(
#         rotation_range=180,
#         width_shift_range=0.1,
#         height_shift_range=0.1,
#         zoom_range=0.1,
#         horizontal_flip=True,
#         vertical_flip=True,
#         preprocessing_function=vgg_pp,
#         fill_mode='nearest')
#
# train_generator = datagen.flow_from_directory(
#         'data/train',
#         target_size=(224, 224),
#         batch_size=batch_size,
#         class_mode='binary',  # this means our generator will only yield batches of data, no labels
#         shuffle=True)
#
# feature_buf = []
# label_buf = []
# for batch in train_generator:
#     print('Batch {}'.format(batch_num_train))
#     # get the bottleneck feature of each batch (an np array)
#     batch_feature = model.predict_on_batch(batch[0])
#     feature_buf.append(batch_feature)
#     label_buf.append(batch[1])
#     batch_num_train -= 1
#     if not batch_num_train:
#         break
#
# print('One batch of features:')
# print(feature_buf[0].shape)
# print('One batch of labels:')
# print(label_buf[0].shape)
#
# bottleneck_features_train = np.concatenate([bf for bf in feature_buf])
# bottleneck_features_train_label = np.concatenate([bl for bl in label_buf])
# print('All features:')
# print(bottleneck_features_train.shape)
# print('All labels:')
# print(bottleneck_features_train_label.shape)
#
# np.save('data/bottleneck_features_train', bottleneck_features_train)
# np.save('data/bottleneck_features_train_label', bottleneck_features_train_label)
#
#
# validation_generator = datagen.flow_from_directory(
#         'data/validation',
#         target_size=(224, 224),
#         batch_size=batch_size,
#         class_mode='binary',  # this means our generator will only yield batches of data, no labels
#         shuffle=True)
#
# feature_buf = []
# label_buf = []
# for batch in validation_generator:
#     print('Batch {}'.format(batch_num_validation))
#     # get the bottleneck feature of each batch (an np array)
#     batch_feature = model.predict_on_batch(batch[0])
#     feature_buf.append(batch_feature)
#     label_buf.append(batch[1])
#     batch_num_validation -= 1
#     if not batch_num_validation:
#         break
#
# print('One batch of features:')
# print(feature_buf[0].shape)
# print('One batch of labels:')
# print(label_buf[0].shape)
#
# bottleneck_features_validation = np.concatenate([bf for bf in feature_buf])
# bottleneck_features_validation_label = np.concatenate([bl for bl in label_buf])
# print('All features:')
# print(bottleneck_features_validation.shape)
# print('All labels:')
# print(bottleneck_features_validation_label.shape)
#
# np.save('data/bottleneck_features_validation', bottleneck_features_validation)
# np.save('data/bottleneck_features_validation_label', bottleneck_features_validation_label)


train_data = np.load('data/bottleneck_features_train.npy')
train_labels = np.load('data/bottleneck_features_train_label.npy')
validation_data = np.load('data/bottleneck_features_validation.npy')
validation_labels = np.load('data/bottleneck_features_validation_label.npy')

# train_labels = np.array(
#     [0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))
# print(train_labels.shape)
#
# validation_data = np.load('bottleneck_features_validation.npy')
# validation_labels = np.array(
#     [0] * 9 + [1] * 8)
# print(validation_labels.shape)
# print(train_data.shape)
#
model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_data, train_labels,
          epochs=50,
          batch_size=batch_size,
          validation_data=(validation_data, validation_labels))
model.save_weights('weights/top_fc_weights_pcm_da')

## model.save('cm_pest')