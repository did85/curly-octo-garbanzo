import numpy as np

from keras.applications import VGG16
from keras import backend
from keras.preprocessing import image
from keras.applications import imagenet_utils as utils
from keras.utils import plot_model
from keras.models import load_model


class Classifier(object):

    def __init__(self, model_type='vgg16'):
        # Instantiate VGG16 model with pre-trained weights from ImageNet
        self.__model_type = model_type
        if model_type == 'vgg16':
            self.__model = VGG16(include_top=True, weights='imagenet')
        elif model_type == 'pcm':
            self.__conv_model = VGG16(include_top=False, weights='imagenet')
            self.__fc_model = load_model('cm_pest')
        self.__target_size = (224, 224)

    def __del__(self):
        backend.clear_session()

    def plot(self, path='model.png'):
        plot_model(self.__model, to_file=path)

    def predict(self, image_path):
        print('Input image: {}'.format(image_path))
        img = image.load_img(image_path, target_size=self.__target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_input(x)

        if self.__model_type == 'vgg16':
            preds = utils.decode_predictions(self.__model.predict(x))

            results = ''
            for p in preds[0]:
                results += '{}: {:.2f}%\n'.format(p[1], float(p[2] * 100))
        elif self.__model_type == 'pcm':
            pred = self.__fc_model.predict(self.__conv_model.predict(x, batch_size=1), batch_size=1)
            if int(pred) == 1:
                results = 'mediterranean_fly'
            else:
                results = 'ambrosia_beetle'
        print(results)
        return results

# test code
if __name__ == "__main__":
    classifier = Classifier(model_type='vgg16')
    classifier.predict('dragonfly/dragonfly1.jpeg')
    classifier.predict('dragonfly/dragonfly2.jpeg')
    classifier.predict('dragonfly/dragonfly3.jpeg')
    # classifier.predict('data/train/mediterranean_fly/0001.jpg')
    # classifier.predict('data/train/mediterranean_fly/0002.jpg')
    # classifier.predict('data/train/mediterranean_fly/0003.jpg')
    # classifier.predict('data/train/mediterranean_fly/0004.jpg')