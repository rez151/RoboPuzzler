from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
import cv2

class Classifire:

    def Classifier(self, img):
        colerType = 1
        image_width = 224
        image_hight = 224

        dense_layers = [0]
        layer_sizes = [32, 32, 64]
        conv_layers = [3]

        image = cv2.resize(img, (image_width, image_hight))
        image = image.astype("float") / 255.0
        # cv2.imshow("resize", image)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        K.set_image_dim_ordering('tf')
        model = Sequential()

        for dense_layer in dense_layers:
            for conv_layer in conv_layers:
                    i = 0
                    model.add(Conv2D(layer_sizes[0], (3, 3), input_shape=(image_width, image_hight, colerType)))
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=(2, 2)))

                    for l in range(conv_layer - 1):
                        model.add(Conv2D(layer_sizes[i+1], (3, 3)))
                        model.add(Activation('relu'))
                        model.add(MaxPooling2D(pool_size=(2, 2)))
                        i+=1

                    model.add(Flatten())
                    for _ in range(dense_layer):
                        model.add(Dense())
                        model.add(Activation('relu'))

                    model.add(Dropout(0.5))
                    model.add(Dense(6))
                    model.add(Activation('softmax'))

        model.load_weights('model/first_try.h5')


        prediction = model.predict(image)

        id = prediction.argmax(1)[0]
        print("prediction in %: "+str(prediction[0][id]*100))

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
    Classifire().Classifier()