from keras.applications import VGG16
from keras import backend
from keras.preprocessing import image
from keras.applications import imagenet_utils as utils
import numpy as np
from keras.utils import plot_model


class Classifier(object):

    def __init__(self, model_type='vgg16'):
        # Instantiate VGG16 model with pre-trained weights from ImageNet
        if model_type == 'vgg16':
            self.__model = VGG16(include_top=True, weights='imagenet')
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
        preds = utils.decode_predictions(self.__model.predict(x))

        results = ''
        for p in preds[0]:
            results += '{}: {:.2f}%\n'.format(p[1], float(p[2] * 100))
        print(results)
        return results

# test code
if __name__ == "__main__":
    classifier = Classifier()
    classifier.predict('dragonfly/dragonfly1.jpeg')
    classifier.predict('dragonfly/dragonfly2.jpeg')
    classifier.predict('dragonfly/dragonfly3.jpeg')