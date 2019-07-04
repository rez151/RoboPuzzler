from keras.preprocessing.image import img_to_array
from keras import models
from keras.models import load_model
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


class Visualization:
    def visualization(self, img):
        model = load_model('model/using/model.h5')
        # model.summary()
        layer_outputs = [layer.output for layer in model.layers[:7]]
        activation_model = models.Model(inputs=model.input,
                                        outputs=layer_outputs)
        activations = activation_model.predict(self.preprocessing(img))

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
                fig1.matshow(layer_activation[0, :, :, i], interpolation='nearest', cmap='viridis')
                fig1.axis('off')
                fig1.axes.get_xaxis().set_visible(False)
                fig1.axes.get_yaxis().set_visible(False)
            plt.savefig(
                'log_img/Visual_ConvLayer_{}.png'.format(
                    number
                )
            )
            plt.show(bbox_inches='tight', pad_inches=0)

    def visualheat(self, imgpfad, id):
        global superimposed_img
        model = load_model('model/model_32_32_64_dense_64.h5')
        # model.summary()
        image = self.preprocessing(imgpfad)
        preds = model.predict(image)
        layer_names = ['conv2d_1', 'conv2d_2', 'conv2d_3']
        layer_sizes = [32, 32, 64]
        j = 0
        for name in layer_names:
            orig_img = image
            superimposed_img = orig_img
            output = model.output[:, np.argmax(preds[0])]
            last_conv_layer = model.get_layer(name)
            grads = K.gradients(output, last_conv_layer.output)[0]
            pooled_grads = K.mean(grads, axis=(0, 1, 2))
            iterate = K.function(
                [model.input],
                [pooled_grads, last_conv_layer.output[0]]
            )
            pooled_grads_value, conv_layer_output_value = iterate([image])
            for i in range(layer_sizes[j]):
                conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
            j += 1
            heatmap = np.mean(conv_layer_output_value, axis=-1)
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)
            if not math.isnan(heatmap[0][0]):
                plt.figure()
                plt.imshow(heatmap)
                plt.axis('off')
                plt.savefig('log_img/Visualization/{}_{}.png'.format(id, name))
                heatmap = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # superimposed_img = heatmap * 0.4 + image
                superimposed_img = heatmap * 0.4 + orig_img
                cv2.imwrite('log_img/Visualization/tmp.png', superimposed_img)
        cv2.imwrite('log_img/Visualization/{}.jpg'.format(id), superimposed_img)

    @staticmethod
    def preprocessing(imgpfad):
        image = cv2.imread(imgpfad)
        image = cv2.resize(image, (224, 224))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        return image


if __name__ == '__main__':
    img_path = 'log_img/Visualization/tmp.png'
    Visualization().visualheat(img_path, 9)