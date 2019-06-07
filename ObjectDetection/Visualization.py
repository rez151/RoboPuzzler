from keras.preprocessing.image import img_to_array
from keras import models
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import cv2


class Visualization:
    def visualization(self):

        model = load_model('model/model.h5')
        model.summary()
        layer_outputs = [layer.output for layer in model.layers[:7]]
        activation_model = models.Model(inputs=model.input,
                                        outputs=layer_outputs)
        activations = activation_model.predict(self.preprocessing())

        n_convs = [0, 3, 6]
        layer_names = []
        for layer in model.layers[:7]:
            layer_names.append(layer.name)

        for number in n_convs:
            fig = plt.figure()
            fig.title = layer_names[number]
            layer_activation = activations[number]
            print(layer_activation.shape)
            size = layer_activation.shape[3]
            for i in range(0, size):
                fig1 = fig.add_subplot(8, 8, i + 1)
                fig1.matshow(layer_activation[0, :, :, i], interpolation='nearest', cmap=None)
                fig1.axis('off')
                fig1.axes.get_xaxis().set_visible(False)
                fig1.axes.get_yaxis().set_visible(False)
            plt.show(bbox_inches='tight', pad_inches=0)

    @staticmethod
    def preprocessing():
        img = cv2.imread('Images/test/Elefant/1.jpg', cv2.IMREAD_GRAYSCALE)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.
        return img


if __name__ == '__main__':
    Visualization().visualization()
