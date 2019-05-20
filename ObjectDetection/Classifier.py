from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
import cv2

class Classifire:

    def Classifier(self,img):
        image_width = 224
        image_hight = 224

        image = cv2.resize(img, (image_width, image_hight))
        image = image.astype("float") / 255.0
        cv2.imshow("resize", image)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)



        K.set_image_dim_ordering('tf')

        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(image_width, image_hight, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))


        model.add(Flatten())
        # model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(6))
        model.add(Activation('softmax'))

        model.load_weights('model/32_32_64_DROP01_DROP05.h5')


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
