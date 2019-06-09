from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2

from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dropout, GlobalAveragePooling2D
from keras.optimizers import SGD


class Classifire:
    def Classifier(self, img):
        image_width = 224
        image_hight = 224
        modelpath = "model/first_try-2.h5"
        image = self.resizeImageToInputSize(img, image_width, image_hight)

        # load model
        model = self.loadModel(modelpath)

        prediction = model.predict(image)

        id = prediction.argmax(1)[0]
        print("prediction " + str(id) + " to: " + str(round((prediction[0][id] * 100), 2)) + "%")

        if id == 0:
            return "Elefant", id
        if id == 1:
            return "Frosch", id
        if id == 2:
            return "Lowe", id
        if id == 3:
            return "Schmetterling", id
        if id == 4:
            return "Sonne", id
        if id == 5:
            return "Vogel", id

    @staticmethod
    def loadModel(path):
        return load_model(path)

    @staticmethod
    def resizeImageToInputSize(img, image_width, image_hight):
        # resize image to match input size
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (image_width, image_hight))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        return image

    def c(self, img):
        # Bilderformat
        colermode = 1
        image_width = 224
        image_hight = 224
        # batch_size, defines the number of samples that will be propagated through the network
        batch_size = 10
        # Sequential, prepare the model to a list of layers
        model = Sequential()
        # Definition for the layout of the network
        dense_layers = [1]
        layer_sizes = [32, 32, 64]
        conv_layers = [3]
        # Coonv2D(number of filters(window_height, window_width), Activationfunction, input_shape)
        # MaxPooling2D(pool_size=(2, 2))
        # loop for network layer

        model.add(Conv2D(32, (3, 3), padding='same', input_shape=(image_width, image_hight, colermode)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        # The next layer is the substitute of max pooling, we are taking a strided convolution layer to reduce the dimensionality of the image.
        model.add(Conv2D(32, (3, 3), padding='same', strides=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        # The next layer is the substitute of max pooling, we are taking a strided convolution layer to reduce the dimensionality of the image.
        model.add(Conv2D(64, (3, 3), padding='same', strides=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (1, 1), padding='valid'))
        model.add(Activation('relu'))
        model.add(Conv2D(6, (1, 1), padding='valid'))
        model.add(GlobalAveragePooling2D())
        model.add(Activation('softmax'))
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

        model.load_weights('model/weights.hdf5')

        image = self.resizeImageToInputSize(img, image_width, image_hight)

        prediction = model.predict(image)

        id = prediction.argmax(1)[0]
        print("prediction " + str(id) + " to: " + str(round((prediction[0][id] * 100), 2)) + "%")

        if (id == 0):
            return "Elefant", id
        if (id == 1):
            return "Frosch", id
        if (id == 2):
            return "Lowe", id
        if (id == 3):
            return "Schmetterling", id
        if (id == 4):
            return "Sonne", id
        if (id == 5):
            return "Vogel", id


if __name__ == '__main__':
    path = 'Images/train/Schmetterling/0.jpg'
    img = cv2.imread(path)
    Classifire().c(img)
