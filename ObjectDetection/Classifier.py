from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
import cv2

class Classifire:

    def Classifier(self,img):
        image = cv2.resize(img, (150, 150))
        image = image.astype("float") / 255.0
        cv2.imshow("Re", image)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        K.set_image_dim_ordering('tf')

        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(150, 150, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(6, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        model.load_weights('model/first_train.h5')

        prediction = model.predict(image)
        id = prediction.argmax(1)[0]

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
